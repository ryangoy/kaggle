import numpy as np
import os
from os.path import join, isfile, isdir
import os.path
from os import listdir, makedirs, remove
import h5py
import pandas as pd
import ast
import re

def save_output(output_dir, output):
  if not isdir(output_dir):
    makedirs(output_dir)

  # TODO: change this to save to csv file
  np.save(output_dir, output)


def load_data(path, labels_path, h5_path, load_from_h5=True, batch_size=32):
  if not load_from_h5 or not isfile(h5_path):
    print('Loading data into hdf5 format...')
    if isfile(h5_path):
      remove(h5_path)
    # Load labels.
    labels_df = pd.read_csv(labels_path)

    # Save to hdf5 file in batches.
    gen = data_gen(path, labels_df, batch_size)
    chunk, labels_chunk = next(gen)
    sample_count = chunk.shape[0]

    with h5py.File(h5_path, 'w') as h5f:
      maxshape = (None,) + chunk.shape[1:]
      labels_maxshape = (None,) + labels_chunk.shape[1:]
      dset = h5f.create_dataset('data', shape=chunk.shape, maxshape=maxshape,
                         chunks=chunk.shape, dtype=chunk.dtype)
      ldset = h5f.create_dataset('labels', shape=labels_chunk.shape, maxshape=labels_maxshape,
                        chunks=labels_chunk.shape, dtype=int)
      dset[:] = chunk
      ldset[:] = labels_chunk
      for chunk, labels_chunk in gen:
        print ('{} samples successfully loaded.'.format(sample_count))
        dset.resize(sample_count + chunk.shape[0], axis=0)
        ldset.resize(sample_count + labels_chunk.shape[0], axis=0)
        dset[sample_count:] = chunk
        ldset[sample_count:] = labels_chunk
        sample_count += chunk.shape[0]
        
  else:
    print('Streaming data from pre-loaded hdf5 file.')
  return h5py.File(h5_path, 'r')

def data_gen(path, labels_df, batch_size):
  data = []
  labels = []
  i = 0
  paths = listdir(path)
  for f in paths:
    fid = f.split('.')[0]
    one_hot = labels_df[labels_df['Id'] == fid]['Label'].tolist()
    # print fid
    # print one_hot
    if one_hot == []:
      continue
    one_hot = one_hot[0]
    one_hot = re.sub('[.]', ',', one_hot)
    one_hot = ast.literal_eval(one_hot)
    data.append(read_data(join(path, f)))
    labels.append(one_hot)
    i += 1
    if i == batch_size or i == len(paths):
      # Yield batch.
      data = np.array(data)
      labels = np.array(labels)
      yield data, labels

      # Reset batch.
      data = []
      labels = []
      i = 0

def read_data(infile):
  """Read any of the 4 types of image files, returns a numpy array of the image contents
  """
  extension = os.path.splitext(infile)[1]
  h = read_header(infile)
  nx = int(h['num_x_pts'])
  ny = int(h['num_y_pts'])
  nt = int(h['num_t_pts'])
  fid = open(infile, 'rb')
  fid.seek(512) #skip header
  if extension == '.aps' or extension == '.a3daps':
      if(h['word_type']==7): #float32
          data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
      elif(h['word_type']==4): #uint16
          data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
      data = data * h['data_scale_factor'] #scaling factor
      data = data.reshape(nx, ny, nt, order='F').copy() #make N-d image
  elif extension == '.a3d':
      if(h['word_type']==7): #float32
          data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
      elif(h['word_type']==4): #uint16
          data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
      data = data * h['data_scale_factor'] #scaling factor
      data = data.reshape(nx, nt, ny, order='F').copy() #make N-d image
  elif extension == '.ahi':
      data = np.fromfile(fid, dtype = np.float32, count = 2* nx * ny * nt)
      data = data.reshape(2, ny, nx, nt, order='F').copy()
      real = data[0,:,:,:].copy()
      imag = data[1,:,:,:].copy()
  fid.close()
  if extension != '.ahi':
      return data
  else:
      return real, imag

def read_header(infile):
  """Read image header (first 512 bytes)
  """
  h = dict()
  fid = open(infile, 'r+b')
  h['filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
  h['parent_filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
  h['comments1'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
  h['comments2'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
  h['energy_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
  h['config_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
  h['file_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
  h['trans_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
  h['scan_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
  h['data_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
  h['date_modified'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 16))
  h['frequency'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['mat_velocity'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['num_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
  h['num_polarization_channels'] = np.fromfile(fid, dtype = np.int16, count = 1)
  h['spare00'] = np.fromfile(fid, dtype = np.int16, count = 1)
  h['adc_min_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['adc_max_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['band_width'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['spare01'] = np.fromfile(fid, dtype = np.int16, count = 5)
  h['polarization_type'] = np.fromfile(fid, dtype = np.int16, count = 4)
  h['record_header_size'] = np.fromfile(fid, dtype = np.int16, count = 1)
  h['word_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
  h['word_precision'] = np.fromfile(fid, dtype = np.int16, count = 1)
  h['min_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['max_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['avg_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['data_scale_factor'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['data_units'] = np.fromfile(fid, dtype = np.int16, count = 1)
  h['surf_removal'] = np.fromfile(fid, dtype = np.uint16, count = 1)
  h['edge_weighting'] = np.fromfile(fid, dtype = np.uint16, count = 1)
  h['x_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
  h['y_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
  h['z_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
  h['t_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
  h['spare02'] = np.fromfile(fid, dtype = np.int16, count = 1)
  h['x_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['y_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['z_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['scan_orientation'] = np.fromfile(fid, dtype = np.int16, count = 1)
  h['scan_direction'] = np.fromfile(fid, dtype = np.int16, count = 1)
  h['data_storage_order'] = np.fromfile(fid, dtype = np.int16, count = 1)
  h['scanner_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
  h['x_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['y_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['z_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['t_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['num_x_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
  h['num_y_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
  h['num_z_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
  h['num_t_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
  h['x_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['y_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['z_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['x_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['y_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['z_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['x_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['y_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['z_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['x_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['y_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['z_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['date_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
  h['time_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
  h['depth_recon'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['x_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['y_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['elevation_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['roll_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['z_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['azimuth_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['adc_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
  h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)
  h['scanner_radius'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['x_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['y_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['z_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['t_delay'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['range_gate_start'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['range_gate_end'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['ahis_software_version'] = np.fromfile(fid, dtype = np.float32, count = 1)
  h['spare_end'] = np.fromfile(fid, dtype = np.float32, count = 10)
  return h