##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

import sys
import copy
import pandas as pd
import time
from pathlib import Path
import argparse

from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TextAbstraction import TextAbstraction

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = 'chapter3_result_outliers.csv'
RESULT_FNAME = 'chapter4_result.csv'

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))



def main():
    print_flags()
    
    start_time = time.time()
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    

    # Let us create our visualization class again.
    DataViz = VisualizeDataset(__file__)
    
    # Compute the number of milliseconds covered by an instance based on the first two rows
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000

    NumAbs = NumericalAbstraction()
    FreqAbs = FourierTransformation()

    if FLAGS.mode == 'aggregation':
        # Chapter 4: Identifying aggregate attributes.

        # Set the window sizes to the number of instances representing 5 seconds, 30 seconds and 5 minutes
        window_sizes = [int(float(5000)/milliseconds_per_instance), int(float(0.5*60000)/milliseconds_per_instance), int(float(5*60000)/milliseconds_per_instance)]

         #please look in Chapter4 TemporalAbstraction.py to look for more aggregation methods or make your own.     
        
        for ws in window_sizes:
                   
            dataset = NumAbs.abstract_numerical(dataset, [FLAGS.aggregatevariable], ws, 'mean')
            dataset = NumAbs.abstract_numerical(dataset, [FLAGS.aggregatevariable], ws, 'std')
      
        DataViz.plot_dataset(dataset, [FLAGS.aggregatevariable, FLAGS.aggregatevariable + '_temp_mean', FLAGS.aggregatevariable + '_temp_std', 'label'], ['exact', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'],
          append_name=FLAGS.plotname)
        print("--- %s seconds ---" % (time.time() - start_time))
  
    if FLAGS.mode == 'frequency':
        # Now we move to the frequency domain, with the same window size.
       
        fs = float(1000)/milliseconds_per_instance
        ws = int(float(10000)/milliseconds_per_instance)
        
        dataset = FreqAbs.abstract_frequency(dataset, FLAGS.usevariables, ws, fs)
        # Spectral analysis.
        DataViz.plot_dataset(dataset, FLAGS.plotvariables + ['label'], 
          (len(FLAGS.plotvariables) + 1)*['like'], len(FLAGS.plotvariables)*['line'] +['points'],
          append_name=FLAGS.plotname)
        print("--- %s seconds ---" % (time.time() - start_time))
  
    if FLAGS.mode == 'final':
        

        ws = int(float(0.5*60000)/milliseconds_per_instance)
        fs = float(1000)/milliseconds_per_instance

        selected_predictor_cols = [c for c in dataset.columns if not 'label' in c]

        dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'mean')
        dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'std')
        # TODO: Add your own aggregation methods here
        
        DataViz.plot_dataset(dataset, ['acc_phone_x', 'gyr_phone_x', 'hr_watch_rate', 'light_phone_lux', 'mag_phone_x', 'press_phone_', 'pca_1', 'label'], ['like', 'like', 'like', 'like', 'like', 'like', 'like','like'], ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points'])

     
        CatAbs = CategoricalAbstraction()
        
        dataset = CatAbs.abstract_categorical(dataset, ['label'], ['like'], 0.03, int(float(5*60000)/milliseconds_per_instance), 2)


        periodic_predictor_cols = ['acc_phone_x'
                                    ,'acc_phone_y','acc_phone_z',
                                    'acc_watch_x','acc_watch_y','acc_watch_z','gyr_phone_x','gyr_phone_y',
                                'gyr_phone_z','gyr_watch_x','gyr_watch_y','gyr_watch_z','mag_phone_x','mag_phone_y','mag_phone_z',
                                'mag_watch_x','mag_watch_y','mag_watch_z']


        
        dataset = FreqAbs.abstract_frequency(copy.deepcopy(dataset), periodic_predictor_cols, int(float(10000)/milliseconds_per_instance), fs)


        # Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be too much alike.

        # The percentage of overlap we allow
        window_overlap = 0.9
        skip_points = int((1-window_overlap) * ws)
        dataset = dataset.iloc[::skip_points,:]


        dataset.to_csv(DATA_PATH / RESULT_FNAME)

        DataViz.plot_dataset(dataset, ['acc_phone_x', 'gyr_phone_x', 'hr_watch_rate', 'light_phone_lux', 'mag_phone_x', 'press_phone_', 'pca_1', 'label'], ['like', 'like', 'like', 'like', 'like', 'like', 'like','like'], ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points'])
        print("--- %s seconds ---" % (time.time() - start_time))
  
if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help= "Select what version to run: final, aggregation or freq \
                        'aggregation' studies the effect of several aggeregation methods \
                        'frequency' applies a Fast Fourier transformation to a single variable \
                        'final' is used for the next chapter ", choices=['aggregation', 'frequency', 'final']) 

    
    parser.add_argument('--plotname', type=str, default='',
                        help='Name appended to the plotted figure.')

    parser.add_argument('--plotvariables', type=str, nargs='+', default=['acc_phone_x_max_freq', 'acc_phone_x_freq_weighted', 'acc_phone_x_pse'],
                        help='Which variables to plot' )
    parser.add_argument('--usevariables', type=str, nargs='+', default=['acc_phone_x'],
                        help='Which variables to use in analysis')
    parser.add_argument('--aggregatevariable', type=str, default='acc_phone_x',
                        help='Name of variable to aggregate.')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()