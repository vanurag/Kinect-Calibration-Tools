#include <opencv2/core/core.hpp>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Boost
#include <boost/math/special_functions/erf.hpp>

static const std::string OPENCV_WINDOW = "Image window";

void depthToCV8UC1(const cv::Mat& in_img, cv::Mat& mono8_img) {
  //Process images
  if(mono8_img.rows != in_img.rows || mono8_img.cols != in_img.cols){
    mono8_img = cv::Mat(in_img.size(), CV_8UC1);}
  cv::convertScaleAbs(in_img, mono8_img);
}

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  float minIr_, maxIr_;
  std::vector<std::pair<double, double>> last_mean_std_8bit;
  double lower_scaling_factor = 0.99; 
  double upper_scaling_factor = 0.99;
  cv::Ptr<cv::CLAHE> clahe_;

  void findMinMax(const cv::Mat_<uint16_t> &ir)
  {
    minIr_ = (float)0xFFFF;
    maxIr_ = 0.0;
    for(size_t r = 0; r < (size_t)ir.rows; ++r)
    {
      const uint16_t *it = ir.ptr<uint16_t>(r);

      for(size_t c = 0; c < (size_t)ir.cols; ++c, ++it)
      {
        minIr_ = std::min(minIr_, (float) * it);
        maxIr_ = std::max(maxIr_, (float) * it);
      }
    }
    //ROS_INFO("in: %d %d\n", minIr_, maxIr_); 
  }

  void findMinMax(const cv::Mat_<float> &ir)
  {
    minIr_ = std::numeric_limits<float>::max();
    maxIr_ = 0.0;
    for(size_t r = 0; r < (size_t)ir.rows; ++r)
    {
      const float *it = ir.ptr<float>(r);

      for(size_t c = 0; c < (size_t)ir.cols; ++c, ++it)
      {
        minIr_ = std::min(minIr_, (float) * it);
        maxIr_ = std::max(maxIr_, (float) * it);
      }
    }
    //ROS_INFO("in: %d %d\n", minIr_, maxIr_);
  }

  // Conversion from power (FLIR data) to Temperature (in Kelvin)
  double powerToKelvin(double power) {
    double slope = 2.58357167114001779457e-07;
    double y_0 = 2.26799217314804718626e+03;
    return sqrt(sqrt(((double)power - y_0) / slope)) - 3;
  }

  // Conversion from power (FLIR image) to Temperature (in Kelvin) image
  cv::Mat powerToKelvinImage(const cv::Mat& power) {
    double slope = 2.58357167114001779457e-07;
    double y_0 = 2.26799217314804718626e+03;

    cv::Mat kelvin = power;
    kelvin.convertTo(kelvin, CV_64F);
    kelvin -= y_0;
    kelvin /= slope;
    sqrt(kelvin, kelvin);
    sqrt(kelvin, kelvin);
    kelvin -= 3.0;
    kelvin.convertTo(kelvin, CV_16U);
    return kelvin;
  }

  //adaptive method with changing thresholds
  void convertTo8bit(const cv::Mat_<uint16_t>& image, cv::Mat& image_8bit,
                     bool update_mean_std, bool power_to_kelvin) {

    int number_of_last_frames = 2;

    image_8bit.create(image.rows, image.cols, CV_8UC1);
    unsigned char* image_8bit_pointer = (unsigned char*)(image_8bit.data);

    // power to kelvin
    cv::Mat kelvin = image;
    unsigned short* kelvin_pointer = (unsigned short*)(kelvin.data);
    if (power_to_kelvin) {
      cv::Mat kelvin = powerToKelvinImage(image);
    }

    int vector_size = last_mean_std_8bit.size();
    if (update_mean_std) {
      cv::Scalar scalar_mean;
      cv::Scalar scalar_deviation;
      cv::meanStdDev(kelvin, scalar_mean, scalar_deviation);
      double deviation = static_cast<double>(scalar_deviation.val[0]);
      double mean = static_cast<double>(scalar_mean.val[0]);

      if (vector_size > number_of_last_frames) {
        last_mean_std_8bit.erase(last_mean_std_8bit.begin());
      }
      last_mean_std_8bit.push_back(std::make_pair(mean, deviation));
    }
    vector_size = last_mean_std_8bit.size();
    
    // Compute average mean and deviation of last number_of_last_frames frames
    double average_mean = 0, average_deviation = 0;
    int frame_id = 0;
    double norm = 0;
    for (auto mean_std_pair : last_mean_std_8bit) {
      double weight = std::exp(-(double) frame_id);
      average_mean += mean_std_pair.first * weight;
      average_deviation += mean_std_pair.second * weight;
      norm += weight;
      ++frame_id;
    }
    average_mean = average_mean / norm;
    average_deviation = average_deviation / norm;
    
    // Scaling limits set to quantiles corresponding
    // to adaptive_threshold_factor_high(low) % confidence.
    double upper_scaling_limit = 
        average_mean - average_deviation * sqrt(2) *
                       boost::math::erfc_inv(2*upper_scaling_factor);
    double lower_scaling_limit = 
        average_mean + average_deviation * sqrt(2) *
                       boost::math::erfc_inv(2*lower_scaling_factor);
    
    // Bouding scaling limits between 0 and 2^16 - 1
    upper_scaling_limit = std::min(upper_scaling_limit, static_cast<double>(pow(2, 16)-1));
    lower_scaling_limit = std::max(lower_scaling_limit, 0.0);

    //ROS_INFO("16-bit to 8-bit conversion: Scaling pixel values from [ %f - %f] -> [ 0 - 255 ]", 
    //          lower_scaling_limit, upper_scaling_limit);  
    for (int j = 0; j < kelvin.rows; ++j) {
      for (int i = 0; i < kelvin.cols; ++i) {
        double temp = kelvin_pointer[kelvin.cols*j + i];

        if (temp > upper_scaling_limit) {
          temp = upper_scaling_limit;
        }
        if (temp < lower_scaling_limit) {
          temp = lower_scaling_limit;
        }

        image_8bit_pointer[image_8bit.cols*j + i] = static_cast<uint8_t>(
            (((temp - lower_scaling_limit) /
              (upper_scaling_limit - lower_scaling_limit)) * (255-50)) + 50);
      }
    }
    // clahe_->apply(image_8bit, image_8bit);
  }


  //adaptive method with changing thresholds
  void convertTo8bit(const cv::Mat_<float>& image, cv::Mat& image_8bit,
                     bool update_mean_std, bool power_to_kelvin) {

    int number_of_last_frames = 2;

    image_8bit.create(image.rows, image.cols, CV_8UC1);
    unsigned char* image_8bit_pointer = (unsigned char*)(image_8bit.data);

    // power to kelvin
    cv::Mat kelvin = image;
    float* kelvin_pointer = (float*)(kelvin.data);

    int vector_size = last_mean_std_8bit.size();
    if (update_mean_std) {
      cv::Scalar scalar_mean;
      cv::Scalar scalar_deviation;
      cv::meanStdDev(kelvin, scalar_mean, scalar_deviation);
      double deviation = static_cast<double>(scalar_deviation.val[0]);
      double mean = static_cast<double>(scalar_mean.val[0]);

      if (vector_size > number_of_last_frames) {
        last_mean_std_8bit.erase(last_mean_std_8bit.begin());
      }
      last_mean_std_8bit.push_back(std::make_pair(mean, deviation));
    }
    vector_size = last_mean_std_8bit.size();

    // Compute average mean and deviation of last number_of_last_frames frames
    double average_mean = 0, average_deviation = 0;
    int frame_id = 0;
    double norm = 0;
    for (auto mean_std_pair : last_mean_std_8bit) {
      double weight = std::exp(-(double) frame_id);
      average_mean += mean_std_pair.first * weight;
      average_deviation += mean_std_pair.second * weight;
      norm += weight;
      ++frame_id;
    }
    average_mean = average_mean / norm;
    average_deviation = average_deviation / norm;

    // Scaling limits set to quantiles corresponding
    // to adaptive_threshold_factor_high(low) % confidence.
    double upper_scaling_limit =
        average_mean - average_deviation * sqrt(2) *
                       boost::math::erfc_inv(2*upper_scaling_factor);
    double lower_scaling_limit =
        average_mean + average_deviation * sqrt(2) *
                       boost::math::erfc_inv(2*lower_scaling_factor);

    // Bouding scaling limits between 0 and 2^16 - 1
    upper_scaling_limit = std::min(upper_scaling_limit, static_cast<double>(pow(2, 16)-1));
    lower_scaling_limit = std::max(lower_scaling_limit, 0.0);

    //ROS_INFO("16-bit to 8-bit conversion: Scaling pixel values from [ %f - %f] -> [ 0 - 255 ]",
    //          lower_scaling_limit, upper_scaling_limit);
    for (int j = 0; j < kelvin.rows; ++j) {
      for (int i = 0; i < kelvin.cols; ++i) {
        double temp = kelvin_pointer[kelvin.cols*j + i];

        if (temp > upper_scaling_limit) {
          temp = upper_scaling_limit;
        }
        if (temp < lower_scaling_limit) {
          temp = lower_scaling_limit;
        }

//        image_8bit_pointer[image_8bit.cols*j + i] = static_cast<uint8_t>(
//            (((temp - lower_scaling_limit) /
//              (upper_scaling_limit - lower_scaling_limit)) * (255-50)) + 50);

        image_8bit_pointer[image_8bit.cols*j + i] = static_cast<uint8_t>(
              (((upper_scaling_limit - temp) /
                (upper_scaling_limit - lower_scaling_limit)) * (255-50)) + 50);
      }
    }
    // clahe_->apply(image_8bit, image_8bit);
  }

  void convertIr(const cv::Mat_<uint16_t> &ir, cv::Mat &grey) {
    const float factor = 255.0f / (maxIr_ - minIr_);
    grey.create(ir.rows, ir.cols, CV_8U);

    #pragma omp parallel for
    for(size_t r = 0; r < (size_t)ir.rows; ++r)
    {
      const uint16_t *itI = ir.ptr<uint16_t>(r);
      uint8_t *itO = grey.ptr<uint8_t>(r);

      for(size_t c = 0; c < (size_t)ir.cols; ++c, ++itI, ++itO)
      {
        *itO = std::min(std::max(*itI - (uint16_t)minIr_, 0) * factor, 255.0f);
      }
    }
    clahe_->apply(grey, grey);
  }

  void convertIr(const cv::Mat_<float> &ir, cv::Mat &grey) {
    const float factor = 255.0f / (maxIr_ - minIr_);
    grey.create(ir.rows, ir.cols, CV_8U);

    #pragma omp parallel for
    for(size_t r = 0; r < (size_t)ir.rows; ++r)
    {
      const float *itI = ir.ptr<float>(r);
      uint8_t *itO = grey.ptr<uint8_t>(r);

      for(size_t c = 0; c < (size_t)ir.cols; ++c, ++itI, ++itO)
      {
        *itO = std::min(std::max(*itI - (float)minIr_, (float)0.0) * factor, 255.0f);
      }
    }
    clahe_->apply(grey, grey);
  }
    
public:
  ImageConverter(std::string input, std::string output)
    : it_(nh_), minIr_(0), maxIr_((float)0x7FFF)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe(input, 1, 
      &ImageConverter::imageCb, this);
    image_pub_ = it_.advertise(output, 1);

    clahe_ = cv::createCLAHE(1.5, cv::Size(32, 32));
  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    std::cout << "ennn" << std::endl;

    if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      try
      {
        std::cout << "ennn1 " << msg->data.size() << " " << msg->height << " " << msg->width << " " << msg->step << std::endl;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
        std::cout << "ennn2" << std::endl;
      }
      catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }
    } else if (msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
      try
      {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
      }
      catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }
    } else {
      ROS_ERROR("Only conversion from 16UC1 and 32FC1 are supported at the moment.");
      exit(1);
    }

    // Convert to mono8
    cv::Mat mono8_img;
    if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      std::cout << "converting 16UC1..." << std::endl;
      cv::Mat_<uint16_t> input_image = cv_ptr->image;
      findMinMax(input_image);
      convertTo8bit(input_image, mono8_img, true, true);
    } else if (msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
      std::cout << "converting 32FC1..." << std::endl;
      cv::Mat_<float> input_image = cv_ptr->image;
      findMinMax(input_image);
      convertTo8bit(input_image, mono8_img, true, true);
    }
//    findMinMax(input_image);
    //ROS_INFO("%d %d\n", minIr_, maxIr_); 
    //convertIr(cv_ptr->image, mono8_img);
//    convertTo8bit(input_image, mono8_img, true, true);
    //cv::resize(mono8_img, mono8_img, cv::Size(739, 415));
    // std::cout << "size: " << mono8_img.rows << ", " << mono8_img.cols << std::endl;
    
    // Output modified video stream
    sensor_msgs::ImagePtr out_msg;
    out_msg = cv_bridge::CvImage(msg->header, "mono8", mono8_img).toImageMsg();
    image_pub_.publish(out_msg);
  }
};

int main(int argc, char** argv)
{
  if (argc != 3) {
    ROS_ERROR("Proper Usage: rosrun image_converter converter <input_topic> <output_topic>");
  }
  ros::init(argc, argv, "image_converter");
  ROS_INFO("Converting and Publishing...");
  ImageConverter ic(argv[1], argv[2]);
  ros::spin();
  return 0;
}
