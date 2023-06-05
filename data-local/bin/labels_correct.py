import shutil
import pandas
import os
import datetime
import re
import time


class Corrector:
    def __init__(self):
        self.original_dir = 'data-local/labels/chexpert'
        self.out_dir = 'data-local/labels/chexpert_correct_labels'
        self.csv_list = []
        self.class_list = ['Atelectasis', 'Cardiomegaly',
                           'Consolidation', 'Edema', 'Pleural Effusion']
        self.correct_data = pandas.read_csv('data-local/chexpert_train.csv')

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

        df.to_csv(self.out_dir + '/' + os.path.basename(csv_path), index=False)

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
    do = Corrector()
    do.run()
    # pass
