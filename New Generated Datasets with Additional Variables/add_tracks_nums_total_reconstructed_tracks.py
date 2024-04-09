import argparse
from ftag.hdf5 import H5Reader
import numpy as np
from numpy.lib.recfunctions import merge_arrays
import h5py

base_labels = {
    0 : 'pile-up',
    1 : 'fake',
    2 : 'primary',
    3 : 'fromB',
    4 : 'fromBC',
    5 : 'fromC',
    6 : 'fromTau',
    7 : 'otherSecondary'
}

def dict_to_structured(jets, d, dtype=None):

    dtype = [(k, dtype if dtype else v.dtype) for k, v in d.items()]
    assert len(set([v.shape for v in d.values()])) == 1, "All arrays must have the same shape"
    arr = np.empty(jets.shape, dtype=list(jets.dtype.descr ) + dtype)
    for field in jets.dtype.names:
        arr[field] = jets[field]
    for k, v in d.items():
        arr[k] = v
    return arr

def run(file, outfile):
    reader = H5Reader(file)
    with h5py.File(file, 'r') as f:
        variables = {k : None for k in f.keys()}
    # variables = {'jets' : None,
    #                     'tracks' : None,
    # }
                        # 'hits' : None}

    with h5py.File(file, 'r') as f:
        dset_shapes = {k : v.shape for k, v in f.items()}
        N = len(f['jets'])
    stream = reader.stream(variables, reader.num_jets, )

    with h5py.File(outfile, 'w') as out:
        dsets = {k : None for k in variables.keys()}
        idx = 0
        
        
        for s in stream:
            # print(type(s))
            tracks = s['tracks']
            
            track_origins = tracks['ftagTruthOriginLabel']
            
            to_store = {f"num_reconstructed_tracks" : None}
            for origin_key, label in base_labels.items():
                num_origin = np.sum(track_origins == origin_key, axis=1)
                to_store[f"num_{label}_tracks"] = num_origin
                if origin_key in [0,1,2,3,4,5,6,7]:
                    if to_store[f"num_reconstructed_tracks"] is None:
                        to_store[f"num_reconstructed_tracks"] = num_origin.copy()
                    else:
                        to_store[f"num_reconstructed_tracks"] += num_origin
            # print({k : v[:10] for k, v in to_store.items()})
            arr = dict_to_structured(s['jets'], to_store, np.int32)
            
            data = {'jets' : arr}
            for k, v in s.items():
                if k != 'jets':
                    data[k] = v
                    
            end = idx + arr.shape[0]

            for k, v in data.items():
                if dsets[k] is None:
                    dsets[k] = out.create_dataset(k, shape=dset_shapes[k], dtype=data[k].dtype)
                    # dsets[k] = out.create_dataset(k, shape=(0,), maxshape=(None,), dtype=v.dtype)
                
                # nsize = [dsets[k].shape[0] + v.shape[0]] +
                dsets[k][idx:end] = v
            idx += v.shape[0]
            print(end, '/', N, ' -> ', round(end / N * 100, 2))

        
        # merged_arr = merge_arrays([tracks, arr], asrecarray=True, flatten=True)
        # print(merged_arr.dtype.names)
        # print(merged_arr[:5])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add data to an HDF5 file.")
    
    # Input files argument (list of files)
    parser.add_argument("--input_files", "-i", type=str, nargs='+', required=True, help="List of paths to the input HDF5 files.")
    
    # Optional output file argument
    parser.add_argument("--output_file", "-o", type=str, default=None, help="Path to the output HDF5 file. This argument is not allowed if multiple input files are provided.")
    
    args = parser.parse_args()
    
    # Check if multiple input files are provided along with an output file
    if len(args.input_files) > 1 and args.output_file is not None:
        raise ValueError("Cannot specify an output file when multiple input files are provided.")
    
    for infile in args.input_files:
        # If no output file is defined or multiple input files are provided, set the output file to be the name of the input file with '_tracknums' at the end
        if args.output_file is None:
            outfile = infile.rsplit('.', 1)[0] + '_tracknums.h5'
        else:
            outfile = args.output_file
        
        run(infile, outfile)


# run("/home/xzcappon/phd/datasets/p5770/mc20/output/pp_output_test_ttbar.h5_resampled_perflav_True_it_1_abs_eta_0.0_2.5_pt_20_250_njets_950406.h5", "/home/xzcappon/phd/datasets/p5770/mc20/output/pp_output_test_ttbar_pt_resampled_withtracknums.h5")
# run("/home/xzcappon/phd/datasets/p5770/mc23/output/pp_output_test_ttbar.h5_resampled_perflav_True_it_1_abs_eta_0.0_2.5_pt_20_250_njets_950406.h5", "/home/xzcappon/phd/datasets/p5770/mc23/output/pp_output_test_ttbar_pt_resampled_withtracknums.h5")
# run("/home/xzcappon/phd/datasets/p5770/mc20/output/pp_output_test_zprime.h5_resampled_perflav_True_it_1_abs_eta_0.0_2.5_pt_250_5000_njets_626267.h5", "/home/xzcappon/phd/datasets/p5770/mc20/output/pp_output_test_zprime_pt_resampled_withtracknums.h5")
# run("/home/xzcappon/phd/datasets/p5770/mc23/output/pp_output_test_zprime.h5_resampled_perflav_True_it_1_abs_eta_0.0_2.5_pt_250_5000_njets_626267.h5", "/home/xzcappon/phd/datasets/p5770/mc23/output/pp_output_test_zprime_pt_resampled_withtracknums.h5")