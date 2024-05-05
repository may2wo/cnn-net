import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import pandas as pd
from csv import writer
    

def main():
    train_filepath = '.../train_data_10.fits'
    galaxy_data = np.array([])
    QSO_data = np.array([])
    star_data = np.array([])
    galaxy_label = np.array([])
    QSO_label = np.array([])
    star_label = np.array([])
    rate = 0.9
    hdulist = fits.open(train_filepath)
    spnum = len(hdulist[0].data) # the 5001st spectra in this fits file
    for sno in range(spnum):
        flux = hdulist[0].data[sno]
        label = hdulist[1].data['label'][sno]
        label = int(label)
        if label == 0:
            if len(galaxy_data) == 0:
                galaxy_data = flux
                galaxy_label = label
            else:
                galaxy_data = np.vstack((galaxy_data, flux))
                galaxy_label = np.vstack((galaxy_label, label))
        elif label == 1:
            if len(QSO_data) == 0:
                QSO_data = flux
                QSO_label = label
            else:
                QSO_data = np.vstack((QSO_data, flux))
                QSO_label = np.vstack((QSO_label, label))
        elif label == 2:
            if len(star_data) == 0:
                star_data = flux
                star_label = label
            else:
                star_data = np.vstack((star_data, flux))
                star_label = np.vstack((star_label, label))
    [galaxy_row, galaxy_col] = galaxy_data.shape
    [QSO_row, QSO_col] = QSO_data.shape
    [star_row, star_col] = star_data.shape
    galaxy_train_num = int(rate * galaxy_row)
    QSO_train_num = int(rate * QSO_row)
    star_train_num = int(rate * star_row)
    galaxy_train_data = galaxy_data[0 : galaxy_train_num, :]
    QSO_train_data = QSO_data[0 : QSO_train_num, :]
    star_train_data = star_data[0 : star_train_num, :]
    galaxy_train_label = galaxy_label[0 : galaxy_train_num, :]
    QSO_train_label = QSO_label[0 : QSO_train_num, :]
    star_train_label = star_label[0 : star_train_num, :]
    galaxy_test_data = galaxy_data[galaxy_train_num : galaxy_row, :]
    QSO_test_data = QSO_data[QSO_train_num : QSO_row, :]
    star_test_data = star_data[star_train_num : star_row, :]
    galaxy_test_label = galaxy_label[galaxy_train_num : galaxy_row, :]
    QSO_test_label = QSO_label[QSO_train_num : QSO_row, :]
    star_test_label = star_label[star_train_num : star_row, :]
    lamost_train_data = np.vstack((galaxy_train_data, QSO_train_data))
    lamost_train_data = np.vstack((lamost_train_data, star_train_data))
    lamost_train_label = np.vstack((galaxy_train_label, QSO_train_label))
    lamost_train_label = np.vstack((lamost_train_label, star_train_label))
    lamost_test_data = np.vstack((galaxy_test_data, QSO_test_data))
    lamost_test_data = np.vstack((lamost_test_data, star_test_data))
    lamost_test_label = np.vstack((galaxy_test_label, QSO_test_label))
    lamost_test_label = np.vstack((lamost_test_label, star_test_label))
    [train_num, channel] = lamost_train_data.shape
    [test_num, channel] = lamost_test_data.shape
    with open('.../lamost_train_data.csv', 'a', encoding='utf-8') as f:
        writer_object = writer(f)
        for sno in range(train_num):
            #serial_no = serial_no + 1
            flux = lamost_train_data[sno]
            flux_list = list(flux)          
            #plt.plot(wavelength,flux)
            #plt.title(f'class:{c[label]}')
            #plt.xlabel('wavelength ({})'.format(f'$\AA$'))
            #plt.ylabel('flux')
            #plt.show()
           
            # Pass the CSV  file object to the writer() function
               
            # Result - a writer object
            # Pass the data in the list as an argument into the writerow() function
            writer_object.writerow(flux_list)
            # Close the file object
        f.close()
    
    with open('.../lamost_train_label.csv', 'a', encoding='utf-8') as f:
        writer_object = writer(f)
        for sno in range(train_num):
            #serial_no = serial_no + 1
            #flux = hdulist[0].data[sno]
            #objid = hdulist[1].data['objid'][sno]
            label = lamost_train_label[sno]
            #wavelength = np.linspace(3900,9000,3000)
            #flux_num = len(flux)
            #arr = np.zeros(flux_num + 1)
            #arr[0] = label
            #arr[1:flux_num+1] = flux
            #label_list = list(int(label))          
            #plt.plot(wavelength,flux)
            #plt.title(f'class:{c[label]}')
            #plt.xlabel('wavelength ({})'.format(f'$\AA$'))
            #plt.ylabel('flux')
            #plt.show()
            
            # Pass the CSV  file object to the writer() function
                
            # Result - a writer object
            # Pass the data in the list as an argument into the writerow() function
            writer_object.writerow([int(label)])
            # Close the file object
        f.close()
    
    with open('.../lamost_test_data.csv', 'a', encoding='utf-8') as f:
        writer_object = writer(f)
        for sno in range(test_num):
            #serial_no = serial_no + 1
            flux = lamost_test_data[sno]
            flux_list = list(flux)          
            #plt.plot(wavelength,flux)
            #plt.title(f'class:{c[label]}')
            #plt.xlabel('wavelength ({})'.format(f'$\AA$'))
            #plt.ylabel('flux')
            #plt.show()
           
            # Pass the CSV  file object to the writer() function
               
            # Result - a writer object
            # Pass the data in the list as an argument into the writerow() function
            writer_object.writerow(flux_list)
            # Close the file object
        f.close()
    
    with open('.../lamost_test_label.csv', 'a', encoding='utf-8') as f:
        writer_object = writer(f)
        for sno in range(test_num):
            #serial_no = serial_no + 1
            #flux = hdulist[0].data[sno]
            #objid = hdulist[1].data['objid'][sno]
            label = lamost_test_label[sno]
            #wavelength = np.linspace(3900,9000,3000)
            #flux_num = len(flux)
            #arr = np.zeros(flux_num + 1)
            #arr[0] = label
            #arr[1:flux_num+1] = flux
            #label_list = list(int(label))          
            #plt.plot(wavelength,flux)
            #plt.title(f'class:{c[label]}')
            #plt.xlabel('wavelength ({})'.format(f'$\AA$'))
            #plt.ylabel('flux')
            #plt.show()
            
            # Pass the CSV  file object to the writer() function
                
            # Result - a writer object
            # Pass the data in the list as an argument into the writerow() function
            writer_object.writerow([int(label)])
            # Close the file object
        f.close()

if __name__ == '__main__':
    main()
