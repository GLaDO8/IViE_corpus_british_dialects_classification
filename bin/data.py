from features import *
from sklearn.model_selection import GridSearchCV, train_test_split

def csv_data_gen(Data, framelength = framelength, framestride = framestride, nfft = nfft, num_fbanks = num_fbanks, n_cep_coeff = n_cep_coeff):
    for nclass in range(1,10):
        for naudio in range(1, 68):
            #calculating mfcc
            rate, data, filename = audio_import(nclass, naudio)
            frames, frmlen = frames_gen(rate, data, framelength, framestride)
            frames = hamming_window(frames, frmlen)
            frame_periodogram = periodogram_gen(frames, nfft)
            fbank = filter_bank_gen(rate, num_fbanks, nfft)
            filter_banks = filtered_frame_gen(frame_periodogram, fbank)
            mfcc = mfcc_gen(filter_banks, n_cep_coeff)

            #calculating delta_coefficients 
#             mfcc = librosa.feature.mfcc(data.astype(float), sr = rate, n_mfcc=12).T
#             mfcc = python_speech_features.base.mfcc(data, rate, winlen = 0.025, winstep = 0.015, nfilt = 40, nfft = 512, numcep = 12, preemph = 0)

            mfcc = mfcc - np.mean(mfcc, axis = 0)
            delta_coef = deltacoeff_gen(mfcc, n_cep_coeff)
            deltadelta_coef = deltadeltacoeff_gen(delta_coef, n_cep_coeff)
            print(mfcc.shape,nclass,naudio)

#             Data = Data.append(pd.Series(np.hstack((np.mean(mfcc, axis = 0), np.mean(delta_coef, axis = 0), nclass))), ignore_index = True)
            
#     Data.columns = ['MFCC_mean' + str(x) for x in range(0, n_cep_coeff)] + ['DEL_mean' + str(x) for x in range(0, n_cep_coeff)] + ['Dialect']
#     return Data

            Data = Data.append(pd.Series(np.hstack(
                (np.mean(mfcc, axis = 0), np.max(mfcc, axis = 0), np.min(mfcc, axis = 0), np.std(mfcc, axis = 0), np.median(mfcc, axis = 0), skew(mfcc, axis = 0), 
                 np.mean(delta_coef, axis = 0), np.max(delta_coef, axis = 0), np.min(delta_coef, axis = 0), np.std(delta_coef, axis = 0), np.median(delta_coef, axis = 0), skew(delta_coef, axis = 0), 
                 np.mean(deltadelta_coef, axis = 0), np.max(deltadelta_coef, axis = 0), np.min(deltadelta_coef, axis = 0), np.std(deltadelta_coef, axis = 0), np.median(deltadelta_coef, axis = 0), skew(deltadelta_coef, axis = 0), nclass))), ignore_index = True)
            
    Data.columns = ['MFCC_mean' + str(x) for x in range(0, n_cep_coeff)] + ['MFCC_max' + str(x) for x in range(0, n_cep_coeff)] + ['MFCC_min' + str(x) for x in range(0, n_cep_coeff)] + ['MFCC_std' + str(x) for x in range(0, n_cep_coeff)] + ['MFCC_median' + str(x) for x in range(0, n_cep_coeff)] + ['MFCC_skew' + str(x) for x in range(0, n_cep_coeff)] + ['DEL_mean' + str(x) for x in range(0, n_cep_coeff)] + ['DEL_max' + str(x) for x in range(0, n_cep_coeff)] + ['DEL_min' + str(x) for x in range(0, n_cep_coeff)] + ['DEL_std' + str(x) for x in range(0, n_cep_coeff)] + ['DEL_median' + str(x) for x in range(0, n_cep_coeff)] + ['DEL_skew' + str(x) for x in range(0, n_cep_coeff)] + ['DELDEL_mean' + str(x) for x in range(0, n_cep_coeff)] + ['DELDEL_max' + str(x) for x in range(0, n_cep_coeff)] + ['DELDEL_min' + str(x) for x in range(0, n_cep_coeff)] + ['DELDEL_std' + str(x) for x in range(0, n_cep_coeff)] + ['DELDEL_median' + str(x) for x in range(0, n_cep_coeff)] + ['DELDEL_skew' + str(x) for x in range(0, n_cep_coeff)] + ['Speaker']
    return Data

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def train_test_gen(filename):
    Data = pd.DataFrame()
    Data = csv_data_gen(Data)
    Data.to_csv(filename, index = False)
    data = pd.read_csv(filename)
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:,-1], test_size = 0.15)
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    return X_train, X_test, y_train, y_test
