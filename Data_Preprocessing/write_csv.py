# Python program to demonstrate writing to CSV  
import csv 
    
def write_csv(X,labels):
    fields = labels
        
    # data rows of csv file 
    rows = X
        
    # name of csv file 
    filename = "university_records.csv"
        
    # writing to csv file 
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        csvwriter.writerows(rows)