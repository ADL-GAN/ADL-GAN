import glob,os,pickle,random,sys
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler,LabelBinarizer
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
# Training Config
checkpoint_path = 'dvector-step160000.pt'
data_dir = ''
normalize_path = ''
val_dir = 'data/1b_all_feature/test'
log_dir = 'exp_1b/logs'
log_step = 100
model_save_step = 10000

batch_size =8
num_iters = 1000000
num_iters_decay = 100000
n_critic = 10
lr = 2e-4
decay=1e-5
num_workers=4

lambda_rec = 1.2
lambda_cls = 1.8
lambda_gp = 0.2


isFile = os.path.isdir(data_dir) and os.path.isdir(normalize_path)

if not isFile:
	raise Exception('Data directory not found. Please check per_precessing.py')

#Preprocessing data

class Singleton(type):

    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super().__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance

def get_label_pretty_name(label):
    if label == 'FIX_walking':
        return 'Walking';
    if label == 'FIX_running':
        return 'Running';
    if label == 'LOC_main_workplace':
        return 'At main workplace';
    if label == 'OR_indoors':
        return 'Indoors';
    if label == 'OR_outside':
        return 'Outside';
    if label == 'LOC_home':
        return 'At home';
    if label == 'FIX_restaurant':
        return 'At a restaurant';
    if label == 'OR_exercise':
        return 'Exercise';
    if label == 'LOC_beach':
        return 'At the beach';
    if label == 'OR_standing':
        return 'Standing';
    if label == 'WATCHING_TV':
        return 'Watching TV'
    
    if label.endswith('_'):
        label = label[:-1] + ')';
        pass;
    
    label = label.replace('__',' (').replace('_',' ');
    label = label[0] + label[1:].lower();
    label = label.replace('i m','I\'m');
    return label

class CommonInfo(metaclass=Singleton):
    """docstring for CommonInfo."""
    def __init__(self, datadir: str):
        super(CommonInfo, self).__init__()
        self.datadir = datadir
    @property
    def speakers(self):
        p = os.path.join(self.datadir, "*")
        all_sub_folder = glob.glob(p)
        all_sub = [s.rsplit('/', maxsplit=1)[1] for s in all_sub_folder]
        all_sub.sort()
        return all_sub
    @property
    def ADL(self,dataset='train'):
        mean_list = ['raw_acc:3d:mean_x','raw_acc:3d:mean_y','raw_acc:3d:mean_z','proc_gyro:3d:mean_x','proc_gyro:3d:mean_y','proc_gyro:3d:mean_z','audio_naive:mfcc0:mean','audio_naive:mfcc1:mean','audio_naive:mfcc2:mean','audio_naive:mfcc3:mean','audio_naive:mfcc4:mean','audio_naive:mfcc5:mean','audio_naive:mfcc6:mean','audio_naive:mfcc7:mean','audio_naive:mfcc8:mean','audio_naive:mfcc9:mean','audio_naive:mfcc10:mean','audio_naive:mfcc11:mean','audio_naive:mfcc12:mean']
        std_list = ['raw_acc:3d:std_x','raw_acc:3d:std_y','raw_acc:3d:std_z','proc_gyro:3d:std_x','proc_gyro:3d:std_y','proc_gyro:3d:std_z','audio_naive:mfcc0:std','audio_naive:mfcc1:std','audio_naive:mfcc2:std','audio_naive:mfcc3:std','audio_naive:mfcc4:std','audio_naive:mfcc5:std','audio_naive:mfcc6:std','audio_naive:mfcc7:std','audio_naive:mfcc8:std','audio_naive:mfcc9:std','audio_naive:mfcc10:std','audio_naive:mfcc11:std','audio_naive:mfcc12:std']

        p = os.path.join(self.datadir, "train/*.csv")
        all_sub_folder = glob.glob(p)
        all_sub_folder.sort()
        if dataset =='train':
            all_sub_folder = all_sub_folder[:len(all_sub_folder)//5*4]
        else:
            all_sub_folder = all_sub_folder[len(all_sub_folder)//5*4:]
        all_speaker = [s.rsplit('/', maxsplit=1)[1].split('.')[0] for s in all_sub_folder]

        X,context = [],[]
        for sub in all_sub_folder:
            df = pd.read_csv(sub)
            #get starting column
            for (ci,col) in enumerate(df.columns):
                if col.startswith('label:'):
                    first_label_ind = ci;
                    break;
                pass;

            df = df.dropna(subset = mean_list)  
            label_names = df.columns[first_label_ind:-1]

            for i, row in df.iterrows():
                for name in label_names:
                    if row[name]==1:
                        X.append(np.array([sub.rsplit('/', maxsplit=1)[1].split('.')[0],row['timestamp']]))
                        context.append(get_label_pretty_name(name[6:]))
                        # speaker.append(sub)

        context_unique = np.sort(np.unique(context))
        X,context = np.array(X).reshape(-1,2),np.array(context)

        if dataset =='train':
            resample = RandomOverSampler()
            X, context = resample.fit_resample(X, np.array(context).reshape(-1,1))

        data_list = [[] for i in range(len(all_speaker))]
        label_list = [[] for i in range(len(all_speaker))]
        for f,c in zip(X,context):            
            [speaker,time] = f
            data_list[all_speaker.index(speaker)].append([speaker,time])
            label_list[all_speaker.index(speaker)].append(c)

        empty_list = []
        for i in range(len(all_speaker)):
            if len(data_list[i]) ==0:
                empty_list.append(i)
        all_speaker = np.delete(np.array(all_speaker),empty_list,axis=0)
        data_list = np.delete(np.array(data_list),empty_list,axis=0)
        label_list = np.delete(np.array(label_list),empty_list,axis=0)

        return context_unique.tolist(),label_list,data_list

class ADLDataset(Dataset):
    """docstring for ADLDataset."""
    def __init__(self):
        super(ADLDataset, self).__init__()
        self.datadir = data_dir
        self.files = file_list
        self.context = context
        self.n_speaker = len(self.files)
        self.encoder = LabelBinarizer().fit(context_unique)
        self.normalizer = {}
        for p in glob.glob(normalize_path+'/*.obj'):
            with open(p,'rb') as file:
                self.normalizer[os.path.basename(p).split('.')[0]] = pickle.load(file)
        assert len(self.normalizer)>0

    def __getitem__(self, idx):
        feature_dir = ['raw_acc','proc_gyro','audio_naive']
        feature_ext = ['.m_raw_acc.dat','.m_proc_gyro.dat','.sound.mfcc']
        rand_ind = np.random.randint(len(self.files[idx%self.n_speaker]))
        p,time = self.files[idx%self.n_speaker][rand_ind]
        label = self.context[idx%self.n_speaker][np.random.randint(len(self.files[idx%self.n_speaker]))]
        time = int(float(time))
        feature_vector = np.zeros((300,19))
        ind=0
        for f_dir,ext in zip(feature_dir,feature_ext):
            filename = os.path.join(self.datadir,f_dir,p,str(time)+ext)
            if f_dir =='audio_naive':
                arr = np.genfromtxt(filename, delimiter=',')[:,:-1]
                arr = np.repeat(arr, 2, axis=0)[:300]
            else:
                arr = np.genfromtxt(filename, delimiter=' ')[:,1:]
                arr = arr[:300]
            feature_dim = arr.shape[1]
            feature_vector[:,ind:ind+feature_dim] = np.pad(arr,((0,300-len(arr)),(0,0)))
            ind += feature_dim
        feature_vector = np.nan_to_num(feature_vector)
        feature_vector = self.normalize(feature_vector,p)
        feature_vector = np.swapaxes(feature_vector,0,1)
        feature_vector = torch.reshape(torch.FloatTensor(feature_vector),(19,300))
        return feature_vector, torch.tensor(idx%self.n_speaker), torch.tensor(context_unique.index(label), dtype=torch.long), torch.FloatTensor(self.encoder.transform([label])[0]), self.normalize_get_param(p)

    def speaker_encoder(self):
        return self.encoder
    def normalize(self,feature,speaker):
        mean_arr,std_arr = self.normalizer[speaker][:,0],self.normalizer[speaker][:,1]
        mean_arr,std_arr = np.tile(mean_arr,(300,1)),np.tile(std_arr,(300,1))
        return (feature-mean_arr)/std_arr

    def normalize_get_param(self,speaker):
        mean_arr,std_arr = self.normalizer[speaker][:,0],self.normalizer[speaker][:,1]
        mean_arr,std_arr = np.tile(mean_arr,(300,1)),np.tile(std_arr,(300,1))
        return [mean_arr,std_arr]
    def __len__(self):
        return 200000000
class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
logger = Logger(log_dir)
context_unique,context,file_list = CommonInfo(data_dir).ADL
context_size = 64 if sys.argv[1]=='subject_transfer' else len(context_unique)

