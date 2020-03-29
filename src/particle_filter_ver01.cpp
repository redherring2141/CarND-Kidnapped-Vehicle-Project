/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using namespace std;
using std::string;
using std::vector;


static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 1000;  // TODO: Set the number of particles
  particles.clear();
  weights.clear();
  particles.resize(num_particles);
  weights.resize(num_particles);


  // Set the sensor noise with normal distributions
  normal_distribution<double> dist_x(0, std[0]);
  normal_distribution<double> dist_y(0, std[1]);
  normal_distribution<double> dist_theta(0, std[2]);

  // Initializae particles
  for (int idx=0; idx<num_particles; idx++)
  {
	  Particle P;
	  P.id = idx;
	  P.x = x;
	  P.y = y;
	  P.theta = theta;
	  P.weight = 1.0;

	  P.x = P.x + dist_x(gen);
	  P.y = P.y + dist_y(gen);
	  P.theta = P.theta + dist_theta(gen);

	  particles.push_back(P);
    weights.push_back(P.weight);
  }

  is_initialized = true;
}



void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate)
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
	
	// Calculate prediction
	for (int idx_p=0; idx_p<num_particles; idx_p++)
	{
    if (fabs(yaw_rate) < 0.0001)
    {
  		particles[idx_p].x = particles[idx_p].x + velocity*delta_t*cos(particles[idx_p].theta);
	  	particles[idx_p].y = particles[idx_p].y + velocity*delta_t*sin(particles[idx_p].theta);
    }
    else
    {
      particles[idx_p].x = particles[idx_p].x + velocity / yaw_rate * (sin(particles[idx_p].theta + yaw_rate*delta_t) - sin(particles[idx_p].theta));
      particles[idx_p].y = particles[idx_p].y + velocity / yaw_rate * (cos(particles[idx_p].theta) - cos(particles[idx_p].theta + yaw_rate*delta_t));
      particles[idx_p].theta = particles[idx_p].theta + yaw_rate * delta_t;
    }
    
    // Add Gaussian noise
    particles[idx_p].x = particles[idx_p].x + dist_x(gen);
    particles[idx_p].y = particles[idx_p].y + dist_y(gen);
    particles[idx_p].theta = particles[idx_p].theta + dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for (unsigned idx_obs=0; idx_obs < observations.size(); idx_obs++)
  {
    // Set the landmark observation as the current observation
    LandmarkObs landmark_obs = observations[idx_obs];

    // Initialize the minimum distance to the maximum value
    // and the map index
    double min_distance = numeric_limits<double>::max();
    int map_idx = -1;

    for (unsigned idx_prd=0; idx_prd < predicted.size(); idx_prd++)
    {
      // Set the landmark prediction as the current prediction
      LandmarkObs landmark_prd = predicted[idx_prd];
      // Calculate the current distance as the distance between observation and prediction
      double cur_distance = dist(landmark_obs.x, landmark_obs.y, landmark_prd.x, landmark_prd.y);

      // Set the predicted landmark to the nearest observed landmark
      if (cur_distance < min_distance)
      {
        min_distance = cur_distance;
        map_idx = landmark_obs.id;
      }
    }

    observations[idx_obs].id = map_idx;
  }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks)
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  // Get the coordinates and yaw angle for each particle
  for (int idx_p=0; idx_p<num_particles; idx_p++)
  {
    double px = particles[idx_p].x;
    double py = particles[idx_p].y;
    double ptheta = particles[idx_p].theta;

    // Generate a vector to store the map landmark locations
    // that will be within sensor range of the particles
    vector<LandmarkObs> predictions;

    // Get the coordinates and indices for each map landmark
    for (unsigned idx_l=0; idx_l<map_landmarks.landmark_list.size(); idx_l++)
    {
      float lx = map_landmarks.landmark_list[idx_l].x_f;
      float ly = map_landmarks.landmark_list[idx_l].y_f;
      int li = map_landmarks.landmark_list[idx_l].id_i;

      //if ( (fabs(lx-px)<=sensor_range) && (fabs(ly-py)<=sensor_range) )
      if (dist(lx, ly, px, py)<=sensor_range)
      {
        predictions.push_back(LandmarkObs{ li, lx, ly });
      }
    }

    // Generate a vector to store the transforemd observations
    vector<LandmarkObs> transformed_observations;

    //cout << observations.size() << endl;

    // Get the coordinates and indices for each observation
    for (unsigned idx_o=0; idx_o<observations.size(); idx_o++)
    {
      double tx = cos(ptheta)*observations[idx_o].x - sin(ptheta)*observations[idx_o].y + px;
      double ty = sin(ptheta)*observations[idx_o].x + cos(ptheta)*observations[idx_o].y + py;
      transformed_observations.push_back(LandmarkObs{ observations[idx_o].id, tx, ty});
    }

    // Perform association for the predictions and transformed observations
    dataAssociation(predictions, transformed_observations);

    // Reinitialize weight
    //particles[idx_p].weight = 1.0;
    double fin_w = 1.0;

    for (unsigned idx_t=0; idx_t<transformed_observations.size(); idx_t++)
    {
      double prx, pry;
      double obx = transformed_observations[idx_t].x;
      double oby = transformed_observations[idx_t].y;
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];

      //int index_associated_prediction = transformed_observations[idx_t].id;

      for (unsigned int idx_pr=0; idx_pr<predictions.size(); idx_pr++)
      {
        //if (predictions[idx_pr].id == index_associated_prediction)
        if (predictions[idx_pr].id == transformed_observations[idx_t].id)
        {
          prx = predictions[idx_pr].x;
          pry = predictions[idx_pr].y;
        }
      }

      // Calculate the weight using multivariate Gaussian function
      //double observation_weight = ( 1/(2*M_PI*std_x*std_y)) * exp( -( pow(prx-obx,2)/(2*pow(std_x,2)) + pow(pry-oby,2)/(2*pow(std_y,2)) ) );
      //double observation_weight = multi_gauss_prob(std_x, std_y, obx, oby, prx, pry);
      fin_w *= multi_gauss_prob(std_x, std_y, obx, oby, prx, pry);
      
      //cout << observation_weight << endl;
      //particles[idx_p].weight = particles[idx_p].weight * observation_weight;
      //particles[idx_p].weight *= multi_gauss_prob(std_x, std_y, obx, oby, prx, pry);
      //particles[idx_p].weight = 1.0;
    }

    cout << fin_w << endl;
    particles[idx_p].weight = fin_w;
    


  }
}

void ParticleFilter::resample()
{
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  vector<Particle> new_particles;

  vector<double> w_list;
  for (int i=0; i<num_particles; i++)
  {
    w_list.push_back(particles[i].weight);
  }

  uniform_int_distribution<int> uni_int_dist(0, num_particles-1);
  auto idx_w= uni_int_dist(gen);

  double max_w = *max_element(w_list.begin(), w_list.end());

  uniform_real_distribution<double> uni_real_dist(0.0, max_w);
  double beta = 0.0;

  for (int i=0; i<num_particles; i++)
  {
    beta += uni_real_dist(gen)*2.0;
    while (beta > w_list[idx_w])
    {
      beta = beta - w_list[idx_w];
      idx_w = (idx_w + 1)%num_particles;
    }
    new_particles.push_back(particles[idx_w]);
  }

  particles = new_particles;
}


void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
