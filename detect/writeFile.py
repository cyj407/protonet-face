import csv
import os
# from os import listdir
# from os.path import isfile, join

label = {
    "\person1":"Justin Bieber",
    "\person2":"Ariana Grande",
    "\person3":"Ed Sheeran",
    "\person4":"Billie Eilish",
    "\person5":"Shawn Mendes",
    "\person6":"Camilia Cabello",
    "\person7":"Troye Sivan",
    "\person8":"Dua Lipa",
    "\person9":"Charlie Puth",
    "\person10":"Selena Gomez",
    "\person11":"Zayn Malik",
    "\person12":"Jennifer Lawrence",
    "\person13":"Donald Trump",
    "\person14":"Dakota Johnson",
    "\person15":"Niall Horan",
    "\person16":"Rihanna",
    "\person17":"Gigi Hadid",
    "\person18":"Jimmy Fallon",
    "\person19":"Ann Hathaway",
    "\person20":"Bradley Cooper",
    "\person21":"Chen Yu Jie"
    # "\\unknown": "unknown"
}

DOC = 'data20_me'

def gen_split_csv():
    path = os.path.join(os.getcwd(), DOC)

    csvlist = []
    for i, data in enumerate(label.items()):
        dirpath = data[0][1:]
        cur_path = os.path.join(os.getcwd(), DOC, dirpath)
        for (d, _, sub) in os.walk(cur_path):
            # print(c)
            for f in sub:
                if str(f).endswith('.jpg'):
                    # print(dirpath + os.sep + str(f))
                    subdirname = data[0]      # \personX
                    filename = os.path.join(subdirname, str(f))
                    
                    # print(label[subdirname])
                    # print(filename)
                    csvlist.append([filename, label[subdirname]])


    print(len(csvlist))
    category = len(label)
    each_class_num = 25  # should be 25's multiple
    train_num = int(each_class_num * 13 / each_class_num)
    test_num = int(each_class_num * 6 / each_class_num)
    val_num = int(each_class_num * 6 / each_class_num)

    # split
    train_rows = []
    test_rows = []
    val_rows = []
    for row in range(category):
        # one class range = row*each_class_num ~ (row+1)*each_class_num
        train_start = row * each_class_num
        test_start = train_start + train_num
        val_start = test_start + test_num
        class_end = (row+1) * each_class_num
        # print("{} {} {} {}".format(train_start, test_start, val_start, class_end))
        train_rows.extend(csvlist[train_start:test_start])
        test_rows.extend(csvlist[test_start:val_start])
        val_rows.extend(csvlist[val_start:class_end])
        # print(csvlist[val_start:class_end])


    # write files
    train_csv = os.path.join(path, 'train.csv')
    val_csv = os.path.join(path, 'val.csv')
    test_csv = os.path.join(path, 'test.csv')

    write_list = [train_csv, val_csv, test_csv]
    write_content = [train_rows, val_rows, test_rows]

    for i in range(len(write_list)):
        with open(write_list[i], 'w', newline='') as csvfile:
            train_writer = csv.writer(csvfile)
            train_writer.writerow(['filename', 'label'])
            train_writer.writerows(write_content[i])
            csvfile.close()

if __name__ == "__main__":
    gen_split_csv()