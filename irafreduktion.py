from pathlib import Path
from pyraf import iraf
import astroscrappy
from astropy.io import fits
import os
iraf.images()
iraf.imutil()
iraf.noao()
iraf.imred()
iraf.ccdred()
iraf.echelle()
iraf.set(clobber = "yes")


def red_echelle_UV(star, inimage, outimage, tharimage):
    
    input_dir = Path.cwd()
    output_dir = input_dir/"spektren_irafreduziert"

    os.chdir(input_dir)
    
    print("********************************************")
    print(" ")
    print("     Automatic data reduction of the")
    print("          TLS Echelle spectra")
    print("       taken with the Andor camera")
    print(" ")
    print("             Version 3.0")
    print(" ")
    print("********************************************")
    print("     Current Frame:     ")
    print(inimage)
    print("********************************************")

    iraf.imcopy(input = inimage, output =  "duma")
    iraf.hedit(images = "duma",fields="DISPAXIS",value = 1,add = "yes")
    iraf.hedit(images = "duma",fields="INSTRUMENT",value="TLS-Echelle",add = "yes")
    print("removing slope")
    iraf.imarith(operand1 = "duma",op = "-",operand2 = "Slope",result = "dumc",pixtype="real",calctyp="real")
    iraf.imdelete(images = "duma")
    print ("removing the BIAS offset:")
    iraf.imarith(operand1 = "dumc",op = "-",operand2 = 300.9, result = "dumd.fits",pixtype="real",calctyp="real")
    iraf.imdelete(images = "dumc")
    print("removing comsmic rays")
    data, header = fits.getdata("dumd.fits", header=True)
    crmask, clean_data = astroscrappy.detect_cosmics(
        data,
        inmask = None,
        sigclip = 3.5,  # Sigma-Schwelle
        sigfrac = 0.3,  # Nachbarschafts-Schwelle
        objlim = 3,  # Schutz vor Linienfraß (2.4 wenn nur eine Aufnahme vorhanden)
        gain = 1.1,  # e-/ADU
        readnoise = 5.0,  # e-
        satlevel = 65535.0,  # Sättigungsgrenze des CCDs
        niter = 8,  # Anzahl der Iterationen
        sepmed = True,  # Speziell für Spektren (1D-Hintergrund)
)
    fits.writeto("dumd2.fits", clean_data, header=header, overwrite=True)
    print("Cosmic Rays erfolgreich mit astroscrappy entfernt!")
    iraf.imdelete(images = "dumd.fits")
    print("extracting image")
    iraf.imcopy(input = "dumd2.fits", output = "dume")
    iraf.imdelete(images = "dumd2.fits")
    print("flat-fielding the image")
    iraf.imarith(operand1 = "dume",op = "/",operand2 = "flat_final",result = "dumf"+"_"+inimage, pixtype="real",calctyp="real")
    iraf.imdelete(images = "dume")
    print(" ")
    print("removing scattered light")
    print(" ")
    iraf.apscatter(input = "dumf"+"_"+inimage, output = "dumi", references ="orders_UV", line=1024, find= "yes", recente = "yes")
    print (" ")
    print("extracting the orders")
    print (" ")
    iraf.apall(input = "dumi", output = "dumj", references = "dumf"+"_"+inimage, background = "none", trace = "no", resize = "no")
    iraf.imdelete(images = "dumi")
    print("")
    print("extracting the thar")
    iraf.imcopy(input = tharimage, output = "dumta")
    iraf.hedit(images = "dumta", fields="DISPAXIS", value = 1, add = "yes")
    iraf.hedit(images = "dumta", fields="DATASEC", add = "no", delete = "yes")
    iraf.hedit(images = "dumta", fields="CCDSEC", add = "no", delete = "yes")
    iraf.hedit(images = "dumta", fields="BIASSEC", add = "no", delete = "yes")
    iraf.hedit(images = "dumta", fields="BIASSEC1", add = "no", delete = "yes")
    iraf.imarith(operand1 = "dumta", op = "-", operand2 = 300.9, result = "dumtb", pixtype= "real", calctyp = "real")
    iraf.imdelete("dumta")
    iraf.imcopy(input = "dumtb", output = "dumtc")
    iraf.imdelete(images = "dumtb")
    print(" ")
    iraf.apall(input = "dumtc",output="dumtd", interactive = "no", references = "dumf"+"_"+inimage, background = "none", weights = "variance", clean = True)
    iraf.imdelete(images = "dumf"+"_"+inimage)
    iraf.imdelete(images = "dumtc")
    iraf.ecreidentify(images = "dumtd", reference = "ThArUV")
    iraf.refspectra(input = "dumj", references = "dumtd", sort = "", group = "")
    iraf.dispcor(input = "dumj", output = "dumk", linearize = "no")
   
    iraf.imdelete(images = "dumtd")
    #iraf.imdelete(images = "dumk")
#     Trick: Die Werte des nicht-rebinnten Spektrums dumj werden       
#     in das File mit dem rebinnten Spektrum (dumk) eingetragen.

    iraf.imreplace(images = "dumk", value = 0.0, lower = "INDEF", upper = "INDEF")
    iraf.imarith(operand1 = "dumk", op = "+", operand2 = "dumj", result = "duml", pixtype = "real", calctyp = "real")
    iraf.splot(images = "duml", new_image = output_dir/f"{outimage}.fits")
    iraf.imdelete(images = "dumj")
    iraf.imdelete(images = "dumk")
    iraf.imdelete(images = "duml")
    print("********************************************")
    print("Wavelength calibrated spectrum created ")
    iraf.imhead(images = output_dir/outimage, longheader = "no")
    print("********************************************")
    #print("Blazekorrektur")
    iraf.gflush()
    iraf.flprcache()
    #iraf.continuum(input = output_dir/outimage, output = f"spektren_blazebereinigt/{outimage}ohneblaze.fits", order = 5, low_reject = 2.0, high_reject = 3.0, niterate = 10)

def main():
    #print("In-image:")
    #in_image = input()
    #print("Out-image:")
    #out_image = input()
    #print("Thar-image:")
    #thar_image = input()

    star = "HD283048"
    input_dir = Path.cwd()/"subdwarfs"/star
    os.chdir(input_dir)

    raw_images = [i.name for i in input_dir.iterdir() if i.is_file() and i.name[:9]=='raw_image']
    raw_images_numbers = []
    for raw_image in raw_images:
        raw_images_numbers.append(int(raw_image[9]))
    
    raw_thar_images = [i.name for i in input_dir.iterdir() if i.is_file() and i.name[:8]=='raw_thar']
    number_raw_images = max(raw_images_numbers)
    
    for i in range(number_raw_images):
        thar_subimages = [image for image in raw_thar_images if int(image[8])==i+1]
        raw_images_subimages = [image for image in raw_images if int(image[9])==i+1]
        #print(thar_subimages)
        #irprint(raw_images_subimages)
        iraf.imcombine(input = ",".join(thar_subimages), output = f'raw_thar{i+1}.fits', combine = "lmedian", reject = "none")
        iraf.imcombine(input = ",".join(raw_images_subimages), output = f'raw_image{i+1}.fits', combine = "average", reject = "crreject")
        
        red_echelle_UV(star, f'raw_image{i+1}.fits', f'out{i+1}', f'raw_thar{i+1}.fits')
        
    #print(raw_images)
    #print(raw_thar_images)
    #red_echelle_UV(star, in_image, out_image, thar_image)

main()