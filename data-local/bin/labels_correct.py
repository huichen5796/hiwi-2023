import shutil
import pandas
import os
import datetime
import re
import time
import numpy as np


class Corrector:
    def __init__(self, original_dir, out_dir, correct_data):
        self.original_dir = original_dir
        self.out_dir = out_dir
        self.csv_list = []
        self.class_list = ['Atelectasis', 'Cardiomegaly',
                           'Consolidation', 'Edema', 'Pleural Effusion']
        self.correct_data = pandas.read_csv(correct_data)

    def creat_out_dir(self):
        print('--------creat_out_dir---------')
        print(f'current path of original_dir is {self.original_dir}.')
        print(f'current path of out_dir is {self.out_dir}.')
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
            print(f"creat {self.out_dir} done!")
        else:
            choice = input(f"{self.out_dir} exists! Empty the folder? (Y/N): ")

            if choice.upper() == "Y":
                shutil.rmtree(self.out_dir)
                os.makedirs(self.out_dir)
                print(
                    f"The folder {self.out_dir} was emptied and recreated successfully!")
            else:
                current_time = datetime.datetime.now()
                formatted_time = current_time.strftime("_%m_%d_%H%M")
                self.out_dir = self.out_dir + formatted_time
                print(
                    f"Folder not be emptied, new data will be writed in {self.out_dir}.")
                os.makedirs(self.out_dir)
                print(f'current path of out_dir is {self.out_dir}.')

    def get_csv_list(self):
        print('--------get_csv_list---------')

        if os.path.isfile(self.original_dir):
            self.csv_list = [self.original_dir]
        elif os.path.isdir(self.original_dir):
            files = os.listdir(self.original_dir)
            pattern = r".*\d{2}\.(csv|txt)$"
            self.csv_list = [os.path.join(self.original_dir, file)
                             for file in files if re.match(pattern, file)]
        print(f'found {len(self.csv_list)} csvs and txts.')

    def write_log(self, log_info):
        with open(self.out_dir + '/log.txt', 'a+') as file:
            file.write(f"{datetime.datetime.now()} {log_info} \n")

    def labels_correct(self, csv_path):
        df = pandas.read_csv(csv_path)
        start = time.perf_counter()
        for i, line in df.iterrows():
            for label_class in self.class_list:
                line_path = df['Path'][i]
                currect_label = self.correct_data[label_class][
                    self.correct_data.index[self.correct_data['Path'] == line_path]].values[0]
                if df[label_class][i] != currect_label:
                    df.loc[i, label_class] = currect_label
                    log_info = f'{csv_path}: change class {label_class} of {line_path} to {currect_label}'
                    self.write_log(log_info)

            # progress bar
            finish = '▓' * int((i+1)*(50/len(df)))
            need_do = '-' * (50-int((i+1)*(50/len(df))))
            dur = time.perf_counter() - start
            print("\r{}/{}|{}{}|{:.2f}s".format((i+1), len(df),
                  finish, need_do, dur), end='', flush=True)
        df_onehot = df.applymap(self.onehot_trans)
        df_onehot.to_csv(self.out_dir + '/' +
                         os.path.basename(csv_path), index=False)

    def onehot_trans(self, label):
        if label == 1.0:
            return [1.0, 0.0, 0.0, 0.0]
        elif label == 0.0:
            return [0.0, 1.0, 0.0, 0.0]
        elif label == -1.0:
            return [0.0, 0.0, 1.0, 0.0]
        elif pandas.isna(label):
            return [0.0, 0.0, 0.0, 1.0]
        else:
            return label

    def correct_in_batch(self):
        for csv_path in self.csv_list:
            print(f'--------start {csv_path}---------')
            self.labels_correct(csv_path)
            print(f'--------end {csv_path}---------')
            print('')

    def run(self):
        print('--------start---------')
        self.creat_out_dir()
        print('')
        self.get_csv_list()
        print('')
        self.correct_in_batch()
        print('')
        print('--------end---------')


if __name__ == '__main__':
    do = Corrector(original_dir='data-local\labels\chexpert',
                   out_dir='data-local/labels/chexpert_correct_labels',
                   correct_data='data-local/chexpert_train.csv')
    do.run()
    # pass
