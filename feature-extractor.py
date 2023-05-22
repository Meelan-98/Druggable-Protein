from csv import writer
from propy import PyPro
import csv

spider_pos_file = open("./original/TS_neg_SPIDER.txt", "r")

lines= 0

tr_neg = 2638
tr_pos = 2446
ts_neg = 474
ts_pos = 448

while lines<ts_neg:
    li = [] 
    name = spider_pos_file.readline().strip()
    sequence = spider_pos_file.readline().strip()
    li.append(name[1:])
    with open('./descriptors/APAAC_TS_neg_SPIDER.csv', 'a', newline='') as f_object:
        # Pass this file object to csv.writer()
        # and get a writer object
        # writer_object = writer(f_object)
        writer_object = csv.writer(f_object)

        DesObject = PyPro.GetProDes(sequence)  # construct a GetProDes object

        # calculate 30 pseudo amino acid composition descriptors
        # paac = DesObject.GetCTD()

        # Amino acid compositon descriptors (20).
        # paac = DesObject.GetAAComp()

        # Type I Pseudo amino acid composition descriptors (default is 30)
        # paac = DesObject.GetPAAC(lamda=10, weight=0.05)

        # Dipeptide composition descriptors (400).
        # paac = DesObject.GetDPComp()

        # Amphiphilic (Type II) Pseudo amino acid composition descriptors.
        paac = DesObject.GetAPAAC(lamda=6, weight=0.5)

        if lines == 0:
            attributes = list(paac.keys())
            attributes.insert(0, "seq_name")
            attributes.append("druggable")
            writer_object.writerow(attributes)

        for t in paac.items():
            li.append(t[1])
        
        #Change to 1 or 0
        li.append("0")
        writer_object.writerow(li)
    
        # Close the file object
        f_object.close()

    lines=lines+2

print("Done")
