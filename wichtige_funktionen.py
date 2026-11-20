import numpy as np
from spectres import spectres
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import pandas as pd

def list_spec_info(file_path, spec_id):

    fit=fits.open(file_path)[0]
    header = fit.header 
    imag = fit.data
    
    #Ich will erst checken wieviele Spektren wir haben
    num_of_spec = int(header['NAXIS2'])
    if not (1 <= spec_id <= num_of_spec):
            raise ValueError(f"Die Spektrum-ID muss zwischen 1 und {num_of_spec} liegen. You gave {spec_id}")

    master_str = ''
    c1 = 0
    while (header.get(f'WAT2_{c1+1:03d}')) is not None:
        master_str += header.get(f'WAT2_{c1+1:03d}') + ' '*(68-len(header.get(f'WAT2_{c1+1:03d}')))
        c1+=1

    master_list = master_str.split('spec')

    if 'multi' in master_list[0]:
        master_list.pop(0)
    if ' ' in master_list:
        master_list.remove(' ')
    if '' in master_list:
        master_list.remove('')

    spec_info = master_list[spec_id-1].split('=')[1].replace('"', '').split()
    
    if len(spec_info) <= 9:
         raise ValueError(f"es liegt nicht non-lineare Dispersion vor, was nicht implementiert ist")

    #Bennenung der Stringinformationen
    ap = float(spec_info[0])#aperture number
    beam = float(spec_info[1])#beam number
    dtype = float(spec_info[2])#dispersion type,2->non-lin, 1->log-lin, 0->lin, -1->no disp
    first_wavelength = float(spec_info[3]) 
    avg_abstand = float(spec_info[4])#lin/log->exact, non-lin->approx
    naxis = int(spec_info[5])#number of pixels
    z = float(spec_info[6])#doppler factor
    aplow = float(spec_info[7])#aperture limits
    aphigh = float(spec_info[8])
   
    w_i = float(spec_info[9])#weight
    Delta_lambda_i = float(spec_info[10])#wavelength offset
    ftype = float(spec_info[11])#type of dispersion function
    if ftype != 1:
        print(f"es liegt eine andere dispersions funktion wie chebyshev polynomial dispersion function vor -->> readmultispec")
        from readmultispec import readmultispec
        data = readmultispec(file_path)
        data1 = data['wavelen'][spec_id-1]
        data2 = data['flux'][spec_id-1]
    else:    
        order = int(spec_info[12]) #Ordnungen der Chebyshev Polynome
        p_min = float(spec_info[13])
        p_max = float(spec_info[14])
        coeff = np.asarray(spec_info[15:], dtype=float) # Coefficients der Ordnungen
        if len(coeff) != order:
            raise AttributeError(f'Error while Reading Data: number of coefficients doesnt fit the order number')
    
        pmiddle = (p_max+p_min)/2
        prange = p_max-p_min
        n = (np.arange(naxis, dtype=float) + 1 - pmiddle)/(prange/2) 
    
        x0 = np.ones(naxis, dtype=float)
        x1 = n
        wave = x0 * coeff[0] + x1 * coeff[1]
        for i in range(2, order):
            x2 = 2 * n * x1 - x0
            wave += x2 * coeff[i]
            x0 = x1
            x1 = x2
    
        wavelengths = w_i*(Delta_lambda_i+wave)/(1+z)
        
        pixel_val = imag[spec_id-1,]
    
        data1 = [wavelengths[i] for i in range(len(wavelengths))]
        data2 = [pixel_val[i] for i in range(len(wavelengths))]
    
        
        #for index, i in enumerate(data2):
            #if i < 0 :
                #data2[index] = 0
                
    return data1, data2

def get_master_lists(file_path, starting_order=1, end_order=0):
    fit=fits.open(file_path)[0]
    header = fit.header 
    
    #Ich will erst checken wieviele Spektren wir haben
    if end_order == 0:
        end_order= int(header['NAXIS2'])
    master_wave= []
    master_flux = []
    for i in range(starting_order, end_order+1):
        wave, flux = list_spec_info(file_path, i)
        master_wave.append(wave)
        master_flux.append(flux)
    return master_wave, master_flux

def calculate_bin_averages(bins, x_values, y_values):
    # Ergebnisliste für die Durchschnittswerte der Bins
    average_x_values = []
    average_y_values = []

    for bin in bins:
        start_index, end_index = bin

        if end_index-start_index <= 50:
            # Hole die entsprechenden Werte aus den Arrays
            x_bin_values = x_values[start_index:end_index+1]
            y_bin_values = y_values[start_index:end_index+1]
    
            # Berechne den Durchschnitt für die X- und Y-Werte
            avg_x = sum(x_bin_values) / len(x_bin_values)
            avg_y = sum(y_bin_values) / len(y_bin_values)
            
            # Füge die Durchschnittswerte zur Ergebnisliste hinzu
            average_x_values.append(avg_x)
            average_y_values.append(avg_y)

        else:
            # teile den bin in zwei teile
            middle = int((start_index+end_index)/2)
            
            # Hole die entsprechenden Werte aus den Arrays
            x_bin_values_1 = x_values[start_index:middle+1]
            x_bin_values_2 = x_values[middle:end_index+1]
            y_bin_values_1 = y_values[start_index:middle+1]
            y_bin_values_2 = y_values[middle:end_index+1]
            
            # Berechne den Durchschnitt für die X- und Y-Werte
            avg_x_1 = sum(x_bin_values_1) / len(x_bin_values_1)
            avg_x_2 = sum(x_bin_values_2) / len(x_bin_values_2)
            avg_y_1 = sum(y_bin_values_1) / len(y_bin_values_1)
            avg_y_2 = sum(y_bin_values_2) / len(y_bin_values_2)
            
            # Füge die Durchschnittswerte zur Ergebnisliste hinzu
            average_x_values.append(avg_x_1)
            average_x_values.append(avg_x_2)
            average_y_values.append(avg_y_1)
            average_y_values.append(avg_y_2)
    
    return average_x_values, average_y_values

def Blaze_estimate_bins(x, y, spec_index):
    spec_id = spec_index + 1
    ''' bins_x = []
    bins_y = []'''
    
    if spec_id == 1:#rauschen,sonst ok
        bin_indices = [[0, 20], [50, 100], [150, 170], [220, 250], [300, 350], [550, 600], [695, 725], [840, 870], [1000, 1150], [1220, 1270], [1550, 1600], [1660, 1800], [1850, 1900], [2000, 2040]]
    
    if spec_id == 2:#!
        bin_indices = [[0, 20], [200, 250], [425, 475], [650, 851], [1100, 1130], [1540, 1590], [1680, 1700], [1950, 2000], [2020, 2040]]
    
    if spec_id == 3:#!
        bin_indices = [[0,20], [40, 80], [250, 300], [350, 510], [550, 650], [670, 720], [900, 945], [960, 1010],[1300,1320], [1950, 2000], [2020, 2040]]
    
    if spec_id == 4:#ok
        bin_indices = [[0, 20], [100, 120], [230, 280],[400, 600], [850, 900], [1180, 1230], [1440, 1490], [1720, 1770], [1830, 1880], [1930, 1980], [2030, 2045]]
    
    if spec_id == 5:#ok
        bin_indices = [[5, 50], [90, 270], [400, 450], [650, 700], [765, 810], [1100, 1150], [1700, 1750], [1800, 1850], [2000, 2025]]
    
    if spec_id == 6:#ok
        bin_indices =[[3, 20], [40, 300], [350, 400],[500, 550], [1030,1050], [1320, 1350], [1500, 1520], [1550, 1600], [1750, 1800], [1900, 1950],  [2020, 2040]]
    
    if spec_id == 7:#cheby nicht gut
        bin_indices = [[3,13], [70,225], [410, 440], [700, 750], [900, 950], [1370, 1390], [1650, 1700], [1900, 1950], [2020, 2040]]
    
    if spec_id == 8:#ok
        bin_indices = [[5, 20], [50, 100], [150, 200], [320, 370], [550, 600], [680, 750], [800, 820], [1230, 1280], [1700, 1720], [1770, 1810], [1900, 1950], [2030, 2040]]
    
    if spec_id == 9:# [490, 540]
        bin_indices = [[5, 15], [30, 160], [370, 400],  [880, 910], [1100, 1120],  [1400, 1450],  [1640, 1700], [1780, 1840], [1900, 2000], [2020, 2040]]
    
    if spec_id == 10:#ok
        bin_indices = [[3, 17], [20, 40], [80, 130], [200, 230], [320, 370], [410, 440], [525, 650], [750, 770], [1000, 1020], [1240, 1260], [1705, 1750], [1850, 2000], [2020,2040]]
    
    if spec_id == 11:#ok
        bin_indices = [[3,7], [10, 60], [200, 230], [310, 350], [480, 530], [935, 955], [1080, 1100], [1420, 1460], [1550, 1600], [1850, 1890], [2030, 2042]]
    
    if spec_id == 12:#ok
        bin_indices = [[4, 14], [20,200], [330, 365], [540, 580], [930, 950], [1080, 1100], [1270, 1300],  [1500, 1650], [1750, 1800], [1900, 2000], [2010, 2025]]
    
    if spec_id == 13:#cheby nicht gut
        bin_indices = [[5, 15], [30, 65], [100, 250], [440, 460], [640, 690],  [930, 950], [1480, 1490], [1700, 1800], [1950, 2000], [2030, 2040]]
    
    if spec_id == 14:#mit peak 4100 im vega spektrum vergleichen
        bin_indices = [[5, 15], [60, 90], [280, 330], [560, 600], [650,750], [1030, 1050], [1160, 1190], [1350, 1400], [1550, 1580], [1650, 1700], [1770, 1800], [1810, 1860], [1970, 2020], [2040, 2045]]
    
    if spec_id == 15:
        bin_indices = [[4, 14], [50, 70], [250, 300], [370, 410], [470, 670],[1060,1080],  [1270, 1290], [1430, 1450], [1490, 1540], [1580, 1680], [1780, 1830], [1880, 1920], [1980, 2010], [2030, 2045]]
    
    if spec_id == 16:
        bin_indices = [[3, 15], [30, 150], [300, 350], [360, 370], [400, 450], [500, 550], [650, 700], [900, 920],   [1310, 1330], [1370, 1420], [1550, 1650], [1850, 1900], [1965, 2000], [2010, 2047]]
    
    if spec_id == 17:#[1600, 1650],
        bin_indices = [[3,13], [140, 170], [330, 380], [450, 480], [535, 563], [630, 950], [1145, 1160],  [1420, 1470],  [1730, 1780], [1900, 1950], [2030, 2045]]
    
    if spec_id == 18:#ok~
        bin_indices = [[5, 70], [100, 150], [250, 300], [550, 600], [735, 780], [1110, 1120], [1300, 1350], [1500, 1600], [1725, 1750], [1780, 1800], [1950, 2000], [2020, 2047]]
    
    if spec_id == 19:#ok
        bin_indices = [[0, 3], [50, 80], [140, 190], [240, 280], [350, 375], [478, 493], [535, 650], [720, 745], [1100, 1120], [1464, 1503], [1640, 1670], [1800, 1950], [2020, 2047]]
    
    if spec_id == 20:#ok vega spektrum vergleichen
        bin_indices = [[3, 40], [150, 200], [350, 380], [518, 560], [590, 635], [820, 840],[1050, 1070],  [1330, 1380], [1500, 1550], [1600, 1650], [1800, 1850], [1990, 2007], [2030, 2045]]
    
    if spec_id == 21:#cheby nicht gut, sonst ok
        bin_indices = [[5, 40], [50, 90], [200, 300], [350, 440], [650,700], [1175, 1185], [1370, 1400],  [1700, 1750], [1920, 1970], [2040, 2045]]
        bin_indices_2 = [[5, 40], [50, 90], [200, 300], [350, 440], [500, 550], [1400, 1450], [1595, 1660], [1780, 1800], [1810, 1880], [1920, 1970], [1990, 2030]]
    
    if spec_id == 22:#ok
        bin_indices = [[3, 13],[30, 100], [150, 260], [300, 550], [640, 680],[1080, 1100], [1440, 1460], [1580, 1620], [1810, 1860], [1900, 1960], [1980, 2020], [2030, 2044]]
    
    if spec_id == 23:#ok
        bin_indices = [[3, 13], [20, 70], [120, 170], [300, 350], [600, 640], [1040, 1070], [1250, 1260],  [1500, 1550], [1600, 1650], [1850, 1900], [1930, 1970], [2030, 2044]]
    
    if spec_id == 24:#ok
        bin_indices = [[0, 5], [50, 125], [180, 210], [300, 350], [500, 550], [680, 730],[1090, 1100],  [1520, 1540], [1800, 1830], [1950, 1970], [2038, 2045]]
    
    if spec_id == 25:#ok
        bin_indices = [[3, 20], [50, 75], [150, 200], [360, 405], [610, 630], [900, 940], [1220,1230], [1510, 1560], [1600, 1760], [1800, 1850], [1900, 1950], [2000, 2018], [2035, 2045]]
    
    if spec_id == 26:#ok
        bin_indices = [[2, 23], [63, 220], [270, 300], [360, 400], [420, 555], [600, 750], [1160,1170], [1590, 1620], [1650, 1690], [1755, 1800], [1850, 1900], [1960, 2000], [2010, 2045]]
    
    if spec_id == 27:#ok
        bin_indices = [[3, 13], [20, 100], [150, 173], [210, 330], [420, 470],[700,720],  [1120, 1140], [1310, 1330], [1560, 1610], [1777, 1810], [1860, 1920], [1990, 2020], [2035, 2046]]
    
    if spec_id == 28:#ok
        bin_indices = [[3, 13], [120, 160], [250, 290], [460, 510], [703, 730],  [960, 970], [1480, 1530], [1720, 1750], [1900, 1950], [1970, 2000], [2040, 2045]]
    
    if spec_id == 29:#ok~
        bin_indices = [[3,20], [130, 180], [350, 400], [550, 600],   [1030, 1050], [1500, 1550], [1750, 1800], [1880, 1950], [2000, 2020], [2030, 2045]]
    
    if spec_id == 30:#ok 
        bin_indices = [[3, 25], [30, 55], [90, 177], [240, 290], [350, 400], [500, 550], [730, 740], [900, 910], [1170, 1190],  [1420, 1460], [1700, 1750], [1800, 1850], [1900, 1950], [2000, 2020]]
    
    if spec_id == 31:#ok
        bin_indices = [[3, 10], [25, 250], [310, 330], [400, 450], [650, 700],[760,780], [1080, 1100], [1330, 1340], [1460, 1500], [1590, 1640], [1705, 1750], [1800, 1900], [1950, 2000], [2040, 2048]]
    
    if spec_id == 32:#ok, cheby ~
        bin_indices = [[3, 20], [100, 150], [310, 330], [400, 450], [700, 750], [1000, 1050], [1130, 1150], [1400, 1500], [1560, 1610], [1650, 1670], [1740, 1760], [1810, 2000], [2020, 2048]]
    
    if spec_id == 33:#ok
        bin_indices = [[3, 20], [30, 150], [290, 330], [350, 450], [550, 590], [740, 815], [1090, 1100],  [1410, 1460], [1630, 1660], [1895, 1920], [2010, 2048]]
        
    if spec_id == 34:#ok
        bin_indices = [[3, 30], [80, 260], [280, 370], [600, 700], [720, 760], [800, 850], [1090, 1100],  [1370, 1420], [1450, 1500], [1570, 1595], [1600, 1630], [1700, 1800], [1850, 1900], [1950, 2040], [2042, 2048]]
    
    if spec_id == 35: #n@j@!
        bin_indices = [[2, 6], [10, 20], [30, 40], [50, 100], [150,200], [230, 250], [300, 400],[420, 470], [580, 600], [680, 700], [1420, 1460], [1600, 1650], [1750, 1950], [1970, 2010], [2020, 2035], [2040, 2048]]
    # [640, 700], [1130, 1190],
    if spec_id == 36:#ok
        bin_indices = [[0,20],[50, 100], [150, 200], [275, 325], [370, 550], [750, 800],[1140,1150],  [1630, 1680], [1710, 1800], [1850, 1900], [1950, 2000], [2020, 2040], [2042, 2048]]
    
    if spec_id == 37:#mh
        bin_indices = [[3, 10], [40, 90], [220, 250], [450, 500], [710, 720], [1060, 1080], [1320, 1340], [1500, 1600], [1850, 1890], [2010, 2020], [2043, 2048]]
    
    if spec_id == 38:
        bin_indices = [[2, 12,], [30, 130], [250, 270], [470, 520], [550, 670], [1150,1155], [1450, 1490], [1580, 1600],  [1780, 1890], [1910, 1940], [1990, 2010], [2020, 2045]]
    
    if spec_id == 39: #mh
        bin_indices = [[3, 20], [40, 80], [178, 230], [248, 351], [660, 680],  [870, 890], [1100,1120],  [1450, 1490], [1570, 1590], [1655, 1670], [1690, 1740], [1900, 1950], [1970, 2000], [2045, 2047]]
    
    if spec_id == 40:#ok
        bin_indices = [[3, 20], [100, 140], [320, 370], [530, 580], [680, 730], [850, 900], [1360, 1400],  [1555, 1600], [1620, 1640], [1805, 1850], [2035, 2046]]
    
    if spec_id == 41:# am rechtem rand komisch aber ok
        bin_indices = [[15, 20], [120, 170], [220, 270], [390, 420], [570, 610], [650, 700], [800, 850], [1040,1060],[1450, 1480], [1710, 1740], [1860, 1890], [1950, 1970], [2020, 2030], [2043, 2045]]
    
    if spec_id == 42:
        bin_indices = [[3, 5], [140, 155], [250, 290], [390, 440], [480, 510], [560, 610], [640, 680],  [1160,1170], [1430, 1445],  [1700, 1740], [1830, 1850], [1970, 2010], [2020, 2045]]
    
    if spec_id == 43:#ok
        bin_indices = [[2, 10], [40, 90],[190, 225], [280, 310], [340, 380], [410, 425], [470, 500], [760, 780] , [1050, 1100], [1470, 1510], [1530, 1580], [1650, 1700], [1780, 1820], [1900, 1960], [2000, 2020], [2030, 2045]]
    
    if spec_id == 44:#ok
        bin_indices = [[3, 10], [60, 100], [210, 240], [280, 320], [370, 420], [470, 500],[600, 630], [700, 720],  [1230, 1270], [1400, 1450], [1500, 1550], [1800, 1850], [1920, 1970], [2030, 2045]]
    
    if spec_id == 45:#ok
        bin_indices = [[3, 8], [30, 80], [120, 170], [320, 350], [600, 620], [850, 890], [1320, 1370], [1415, 1460], [1800, 1815], [1930, 1945], [2020, 2045]]
    
    if spec_id == 46:#ok
        bin_indices = [[3, 10], [30, 50], [190, 200], [235, 250], [300, 330], [445, 465], [610, 630], [800, 850], [1150, 1160], [1480, 1520], [1750, 1800], [1900, 1950], [2000, 2020], [2040, 2045]]
    
    if spec_id == 47:
        bin_indices = [[5, 15], [20, 70], [90, 110], [120, 150], [185, 235], [290, 450], [550, 600], [700, 750], [820, 880], [920, 970], [1045, 1150], [1170, 1300], [1370, 1460], [1520, 1560], [1570, 1630], [1640, 1710], [1740, 1810], [1870, 1960], [2030, 2045]]
    
    
    
    
    
    x_spline, y_spline = calculate_bin_averages(bin_indices, x, y)

    #print(x_spline)

    cs = CubicSpline(np.array(x_spline), np.array(y_spline))
    return cs
    
def delta(flux, neighbor):
    flux = list(flux)
    lower = [flux[0]] * neighbor
    upper = [flux[-1]] * neighbor
    master = lower + flux + upper
    out = []
    for f in flux:
        out.append(f-0.5*(master[master.index(f, neighbor, -neighbor)-neighbor] + master[master.index(f, neighbor, -neighbor)+neighbor]))
    out = np.array(out)
    return out

def gauss_poly(x, A, mu, sigma, a0):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + a0# + a1 * x + a2 * x ** 2

def remove_outliers(array, x0, sigma):
    out = []
    for i in array:
        if x0-2*sigma <= i <= x0+2*sigma:
            out.append(i)
    return out

def get_sorted_list_and_per(list1, per=True):
    L = [(list1[i],i) for i in range(len(list1))]
    L.sort()
    sorted_list, permutation = zip(*L)
    sorted_list = list(sorted_list)
    if per == True:
        return sorted_list, permutation
    if per == False:
        return sorted_list

def sort_with_per(to_sort_list, permutation):
    sorted_list = []
    for index2 in permutation:
        sorted_list.append(to_sort_list[index2])
    return sorted_list

def get_signal_to_noise(flux, **kwargs):
    neighbor = kwargs.get('neighbor', 2)
    conf_level = kwargs.get('conf_level', 0)
    if conf_level != -1 and conf_level != 0 and conf_level != 1 and conf_level != 2:
        raise ValueError("Usage error in get_signal_to_noise: Invalid value for qualifier 'conf_level'")
    ndist = delta(flux, neighbor)
    if conf_level != -1:
        key_list = ['noise_min', 'noise', 'noise_max', 'snr_min', 'snr', 'snr_max']
    else:
        key_list = ['noise', 'snr']
    out = dict.fromkeys(key_list, [])
    if len(np.nonzero(ndist)) == 0:
        out['noise'] = 0
        out['snr'] = np.inf
        if conf_level != -1:
            out['noise_min'] = 0
            out['noise_max'] = 0
            out['snr_min'] = np.inf
            out['snr_max'] = np.inf
    else:
        ndist_filtered = remove_outliers(ndist, np.average(ndist), np.std(ndist))
        hist, bin_edges = np.histogram(ndist_filtered, bins=200)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        hist_errors = np.sqrt(hist)
        hist_errors[hist_errors == 0] = 1

        #pos_shift = abs(2 * bin_centers[0])
        #bin_centers += pos_shift

        #obs = define_couts(hist)

        p0 = [len(ndist_filtered), np.average(ndist_filtered), np.std(ndist_filtered), 0]
        bounds = ([-np.inf, -np.inf, 0.1*np.std(ndist_filtered), -np.inf], [np.inf, np.inf, 10*np.std(ndist_filtered)+10**-9, np.inf])

        para, cov = curve_fit(gauss_poly, bin_centers, hist, p0=p0, bounds = bounds, sigma = hist_errors)
#, sigma = hist_errors
        conv_factor = np.sqrt(1.5)
        f_med  = np.median(flux)
        noise = para[2]/conv_factor
        out['noise'] = noise
        out['snr'] = f_med/noise
        if conf_level != -1:
            perr = np.sqrt(np.diag(cov))
            noise_min = (para[2]-perr[2])/conv_factor
            out['noise_min'] = noise_min
            noise_max = (para[2]+perr[2])/conv_factor
            out['noise_max'] = noise_max
            out['snr_min'] = f_med / noise_min
            out['snr_max'] = f_med / noise_max
            
    return out

def optimize_wavegrid_echelle(l, f, res):
    lambda_optimal=[]
    for i in range(len(l)):
        lambda_optimal.append(l[0]*((4*res+1)/(4*res-1))**i)
    sorted_l, per = get_sorted_list_and_per(l)
    sorted_f = sort_with_per(f, per)
    
    flux_optimal = spectres(np.array(lambda_optimal), np.array(sorted_l), np.array(sorted_f), fill = 0, verbose=False)
    lambda_optimal = np.array(lambda_optimal)
    return lambda_optimal, flux_optimal

def get_signal_to_noise_curve(l, f, data_points=3000, conf_level=0):
    len_l = len(l)
    if conf_level != -1:
        out = {'l':[], 'noise_min':[],'noise':[],'noise_max':[],'snr_min':[],'snr':[],'snr_max':[]}
    else:
        out = {'l':[],'noise':[],'snr':[]}
    index = 0
    if data_points > len_l:
        data_points = len_l
    data_points -= 1 #conversion to indices
    while index + data_points <= len_l - 1:
        #slice = [index:index+data_points:1]
        flux_temp = f[index:index+data_points]
        out_temp = get_signal_to_noise(flux_temp, conf_level = conf_level)
        for key in out_temp.keys():
            out[key].append(out_temp.get(key))
        out['l'].append(l[int(index+data_points/2)])
        index += int(data_points/2)
    #the last interval needs special treatment
    #slice =  [len-1-data_points : len-1 : 1]
    flux_temp = f[len_l-1-data_points:len_l-1]
    out_temp = get_signal_to_noise(flux_temp, conf_level = conf_level)
    for key in out_temp.keys():
        if key not in out or not isinstance(out[key], list):
            out[key] = []
        out[key].append(out_temp.get(key))
        out[key] = np.array(out[key])
    out['l'].append(l[int(len_l-data_points/2)])
    out['l'] = np.array(out['l'])
    return out

def coadd(file_path):
    header = fits.open(file_path)[0].header
    n_orders = header['NAXIS2']
    slit = header['CO-SLIT']
    l_in = [list_spec_info(file_path, i)[0] for i in range(1,n_orders+1)]
    f_in = [list_spec_info(file_path, i)[1] for i in range(1,n_orders+1)]
    u = []
    for index, order_l in enumerate(l_in):
        u.extend(order_l)
        f_in[index] /= np.median(f_in[index])
    u = np.array(sorted(u))
        
    if slit.strip() == '0.90mm':
        res = 35000 #source: kso.tls-tautenburg.de/TLS/index.php?id=31 // nachgeprüft
    elif slit.strip() == '1.00mm' or slit.strip() == '1.0mm':
        res = 32000 # nachgeprüft
    elif slit.strip() == '0.52mm':
        res = 50000
    else:
         raise ValueError(f"Spaltbreite von {slit} muss noch implementiert werden")
        
    l_new = [optimize_wavegrid_echelle(u, np.interp(u, l_in[i],f_in[i], left = 0, right = 0), res)[0] for i in range(n_orders)]
    f_new = [optimize_wavegrid_echelle(u, np.interp(u, l_in[i],f_in[i], left = 0, right = 0), res)[1] for i in range(n_orders)]

    len2 = len(l_new[0])
    f = []
    norm = []
    for i in range(n_orders):
        l_new[i] = l_new[i][f_new[i]!=0]
        f_new[i] = f_new[i][f_new[i]!=0]
        
        snr = get_signal_to_noise_curve(l_new[i], f_new[i], data_points = 200, conf_level = -1 )
        weight = (np.interp(l_new[i], snr['l'], snr['snr']))**2
        norm.append(weight)
        f.append(weight*f_new[i])
    for index, li in enumerate(norm):
        f[index] /= li

    return l_new, f

def coadd_list(l_in, f_in, file_path, starting_order=1, end_order=0):
    header = fits.open(file_path)[0].header
    if end_order == 0:
        n_orders = header['NAXIS2']
    else:
        n_orders = end_order-starting_order
    slit = header['CO-SLIT']
    
    u = []
    for index, order_l in enumerate(l_in):
        u.extend(order_l)
        f_in[index] /= np.median(f_in[index])
    u = np.array(sorted(u))
        
    if slit.strip() == '0.90mm':
        res = 35000 #source: kso.tls-tautenburg.de/TLS/index.php?id=31 // nachgeprüft
    elif slit.strip() == '1.00mm' or slit.strip() == '1.0mm':
        res = 32000 # nachgeprüft
    elif slit.strip() == '0.52mm':
        res = 50000
    else:
         raise ValueError(f"Spaltbreite von {slit} muss noch implementiert werden")
        
    l_new = [optimize_wavegrid_echelle(u, np.interp(u, l_in[i],f_in[i], left = 0, right = 0), res)[0] for i in range(n_orders)]
    f_new = [optimize_wavegrid_echelle(u, np.interp(u, l_in[i],f_in[i], left = 0, right = 0), res)[1] for i in range(n_orders)]

    len2 = len(l_new[0])
    f = []
    norm = []
    for i in range(n_orders):
        l_new[i] = l_new[i][f_new[i]!=0]
        f_new[i] = f_new[i][f_new[i]!=0]
        
        snr = get_signal_to_noise_curve(l_new[i], f_new[i], data_points = 200, conf_level = -1 )
        weight = (np.interp(l_new[i], snr['l'], snr['snr']))**2
        norm.append(weight)
        f.append(weight*f_new[i])
    for index, li in enumerate(norm):
        f[index] /= li

    return l_new, f

def group_average(group):
    flux = np.array(group["flux"]) 
    snr = np.where(np.array(group["snr"]) == 0, 0.01, np.array(group["snr"])*gauss_poly(group["index_in_order"], 0.1, group["len_of_order"]/2, group["len_of_order"]/20, 0))
    return np.average(flux, weights = snr)

def weighted_average(group):
    # Berechne den gewichteten flux
    weighted_flux = group_average(group)
    
    # Optional: Berechnung des gewichteten snr (oder andere Statistiken)
    avg_snr = group['snr'].mean()
    
    return pd.Series({'weighted_flux': weighted_flux, 'avg_snr': avg_snr})

def calc_snr_pointwise(intensity, bin_size):
    snr = np.ones(len(intensity))
    for i in range(len(intensity)):
        bin_slice = intensity[i:i + bin_size]
        if np.std(bin_slice) != 0:
            snr[i] = np.mean(bin_slice) / np.std(bin_slice)
        else:
            snr[i] = 0
    return snr

def merge_orders(file_path, **kwargs):
    """
    Kombiniert mehrere Spektren aus einer FITS-Datei, um einen gewichteten Durchschnitt von 
    Flux und Wellenlängen zu berechnen.

    Diese Funktion liest Spektren aus der angegebenen FITS-Datei, entfernt ungültige Werte und 
    berechnet für jeden Wellenlängenwert den gewichteten Durchschnitt des Flux. 
    Zusätzlich werden Signal-Rausch-Verhältnisse (SNR) für die Fluxwerte berechnet. Vor der 
    Berechnung wird der Flux durch eine Schätzung des Blaze-Profils normalisiert.

    Parameters
    ----------
    file_path : str
        Der Pfad zur fertig reduzierten, also wellenlängenkalibrierten, FITS-Datei, die die Spektrendaten enthält.

    Returns
    -------
    tuple
        Ein Tupel, das die Wellenlängen und die entsprechenden gewichteten Fluxwerte enthält.
        - wavelengths : list
            Eine Liste der Wellenlängenwerte.
        - weighted_flux : list
            Eine Liste der gewichteten Fluxwerte, die für jede Wellenlänge berechnet wurden.

    Notes
    -----
    Die Funktion verwendet eine Bin Größe von 20 für die SNR-Berechnung. 
    Ungültige Fluxwerte (z.B. gleich 0) werden entfernt, und die Wellenlängen- und Fluxarrays 
    werden um die Randwerte gekürzt, um die Datenqualität zu verbessern. 
    Es wird angenommen, dass die Funktion `Blaze_estimate_vega` eine Funktion ist, 
    die für die Entfernung der Blazefunktion verwendet wird.

    Example
    --------
    >>> wavelengths, fluxes = merge_orders('path/to/spectrum_file.fits')
    """
    bin_size = 20

    master_wave, master_flux = get_master_lists(file_path, **kwargs)
    
    #BLAZE-FUNKTION
    for i in range(len(master_flux)):
        master_flux[i] /= Blaze_estimate_bins(master_wave[i], master_flux[i], i)(master_wave[i])
    #BLAZE-FUNKTION 
    
    master_wave, master_flux = coadd_list(master_wave, master_flux, file_path, **kwargs)
    
    for i in range(len(master_wave)):
        master_wave[i] = master_wave[i][master_flux[i] != 0]
        master_flux[i] = master_flux[i][master_flux[i] != 0]
        master_wave[i] = master_wave[i][1:-1]
        master_flux[i] = master_flux[i][1:-1]

    data = {
        "wavelength": [],
        "flux": [],
        "snr": [],
        "order_index": [],
        "index_in_order": [],
        "len_of_order": []
    }
    
    # Sammle alle Wellenlängen, fluxe und SNR in einem DataFrame
    for order_index, (order_wave, order_flux) in enumerate(zip(master_wave, master_flux)):
        snr = calc_snr_pointwise(order_flux, bin_size)
        data["wavelength"].extend(order_wave)
        data["flux"].extend(order_flux)
        data["snr"].extend(snr)
        data["order_index"].extend([order_index] * len(order_wave))
        data["index_in_order"].extend(list(range(len(order_wave))))
        data["len_of_order"].extend([len(order_wave)] * len(order_wave))
        
    df = pd.DataFrame(data)
    combined_flux = []
    combined_wave = []
        
        # Gruppiere nach Wellenlänge, um den flux von mehrfach vorkommenden Wellenlängen zu berechnen
    grouped = df.groupby(["wavelength"], as_index=False).apply(weighted_average, include_groups=False).reset_index()
    return grouped["wavelength"], grouped["weighted_flux"]

