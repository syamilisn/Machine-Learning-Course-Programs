import csv
from pathlib import Path
class wc:    
    def write_csv(X,labels,file):
        fields = labels       
        # data rows of csv file 
        rows = X            
        # name of csv file 
        filename = file            
        # writing to csv file 
        #path = Path('B:\Mach_learn_course\Datasets\CKD_Datasets')
        location = input("Enter destination path: ")
        path = Path(location)
        #create directory if not exists
        if not path:
            path.mkdir(parents=True)
        fpath = (path / filename).with_suffix('.csv')
        with fpath.open(mode='w+') as csvfile:
        #with open(filename, 'w') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
                
            # writing the fields 
            csvwriter.writerow(fields) 
                
            # writing the data rows 
            csvwriter.writerows(rows)