# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 09:46:29 2017

@author: t.vorauer
"""
#%%
import numpy as np
from scipy.signal import argrelextrema
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from os import listdir
from os.path import isfile, join
import time
from skimage import measure
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage.morphology import erosion
import re
import csv
from numba import jit
from skimage import color
#%%
def binmasks(elements, andor = 'and'):
    from functools import reduce
    if andor == 'and': return reduce(np.bitwise_and, elements)
    if andor == 'or' : return reduce(np.bitwise_or, elements)
    else:
        print('andor should be either and/or not {}'.format(andor))
        return
#%%
def arrselect(arr, val = 0, bse = 'b'):
    if bse == 'b': return arr[arr>val]
    if bse == 's': return arr[arr<val]
    if bse == 'e': return arr[arr==val]
    else:
        print('bse should be b (bigger), s (smaller) or e (equal), not {}'.format(bse))
        return
#%%
def loadnpy(path):
    return np.load(path+os.sep+[i for i in files(path) if 'npy' in i][0])
#%%
def seggymacsegseg(dat, thh = 18, inter = 100, name = 'test', plot = True, **kwargs):
    hist, bins = np.histogram(dat.flatten(), bins = len(np.unique(dat)[1:]), range= (np.unique(dat)[1], dat.max()), **kwargs)
    part, bini = map(np.array_split, [interpol(hist, inter), bins[1:]], [20, 20])
    part = part[1:-1]
    bini = bini[1:-1]
    fit = list(map(np.polyfit, bini, part, [1]*len(bini)))
    ff = list(map(np.poly1d, fit))
    bla = min(int(np.log(np.mean(i))) for i in part)
    order = np.array([int(len(bini[i])/(int(np.log(np.mean(j)))+np.abs(bla)+1)) for i,j in enumerate(part)])
    order[order==0]=len(bini[0])//2
    linfit = np.array([ff[i](bini[i]) for i in range(18)])
    pieces = np.array(part)-linfit
    extrema = [argrelextrema(pieces[i], np.less, order = order[i])[0].tolist()+argrelextrema(pieces[i], np.greater, order = order[i])[0].tolist() for i,j in enumerate(part)]
        
    threshi = [bini[i][extrema[i]] for i in range(18)]
    flat_list = sorted([item for sublist in threshi for item in sublist])
    
    if plot is True:
        segplot(dat, thresh = flat_list[thh], name = name)
    return flat_list
#%%
def lolconv(lol, array = True):
    if array: return np.array([i for sublist in lol for i in sublist])
    return [i for sublist in lol for i in sublist]
#%%
def dicconv(lis):
    import copy
#    dic0 = {}
    dic0 = copy.deepcopy(lis[0])
    keys = dic0.keys()
    for i in lis[1:]:
        for j in keys:
            dic0[j]+=i[j]
    return dic0
#%%
def segplot(dat, thresh, name = 'test'):
    fig, ax = plt.subplots(1,3, figsize = (22,12), num = 'threshtest'+name)
    ax[0].imshow(intensity2height(dat, 65535,0,'uint16'), cmap = 'gray')
    if np.array(thresh).size == 1:
        ax[1].imshow(mark_boundaries(intensity2height(dat, 65535,0,'uint16'), nd.label(dat>thresh)[0]))
        ax[2].imshow(color.label2rgb(nd.label(dat>thresh)[0], intensity2height(dat, 65535,0,'uint16'), kind='overlay', bg_label = 0))
    else:
        mask = np.bitwise_and(dat>thresh[0], dat<thresh[1])
        ax[1].imshow(mark_boundaries(intensity2height(dat, 65535,0,'uint16'), nd.label(mask)[0]))
        ax[2].imshow(color.label2rgb(nd.label(mask)[0], intensity2height(dat, 65535,0,'uint16'), kind='overlay', bg_label = 0))
    for k in ax: k.axis('OFF')
#%%
def numericalSort(value):
    
    '''Key argument for sorted, (sorted(list, key = numericalSort))'''
    def __name__(self):
        self.name = 'numericalSort'
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
#%%
def bb2coords(bbox):
        if np.shape(bbox)[-1] == 6:
            return (slice(bbox[0],bbox[3]),slice(bbox[1],bbox[4]),slice(bbox[2],bbox[5]))
        elif np.shape(bbox)[-1] == 4:
            return (slice(bbox[0],bbox[2]),slice(bbox[1],bbox[3]))
#%%
def sizefilter(arr, size):
    from skimage.segmentation import relabel_sequential
    props = measure.regionprops(arr)
    test2 = np.array([l.bbox for l in props])
    flase = np.array([min(l[-arr.ndim:]-l[:arr.ndim])<=size for l in test2])
    bla = np.append(True,flase)[arr]
    arr[bla]=0
    return relabel_sequential(arr)[0]
#%%
def writedict(path,mydict, name = 'info'):
    
    '''
    writedict(path, mydict)
    
        path:   string; path where the dictionary will be saved; 
        mydict: dict; Dictionary that will be saved
        name:   string; the file will be named 'info.csv' by default
    '''
    with open(path+os.sep+name+'.csv', 'w') as f:
        w = csv.writer(f)
        for key, value in mydict.items():
            w.writerow([key, value])
   #%%         
#@jit            
def interpol(ar, iteration = 10):
    
    ar = np.array(ar).astype(float)
    first = ar[0]
    last = ar[-1]
    
    for j in range(iteration):
        ar = [(ar[i]+ar[i+1])/2.0 for i in range(len(ar)-1)]
        if not j%2:
            ar = [first]+ar
        else:
            ar.append(last)
    return np.array(ar)
#%%
#@jit            
def intensity2height(img, hmax = 1, hmin = 0, typ = 'float32'):
    
    '''
    intensity2height(img, hmin, hmax)
    
        img:    ndarray; array that will be linarly mapped:
                [img.min, img.max] -> [hmin, hmax]
        hmax:   max value of new range [hmin, hmax],
                optionally, default: 1
        hmin:   min value of new range [hmin, hmax], 
                optionally, default: 0
    '''
    if img.max() == img.min(): return img
    return ((hmax-hmin)*1.0/(img.max()-img.min())*(img-img.min()) + hmin).astype(typ)
#%%
def circle_mask(ar, radii):
    
    from skimage.transform import hough_circle
    from skimage.draw import circle
    
    '''
    circle_mask(ar, radii)
    
        ar:     ndarray; ar where circumferce of circle is 1(true), 
                else 0(false)
        radii:  int or list; int or list of radii to look for
        
    return: mask with filled circle, (cx, cy, r) of circle
    '''

    hough_res = hough_circle(ar, radii)
    
    rr, r, c = np.unravel_index(np.argmax(hough_res), hough_res.shape)
    
    im = np.zeros_like(ar)
    ra, cc = circle(r,c,radii[rr])
    
    im[ra,cc] = 1
      
    return im.astype(bool), (r,c,radii[rr])
#%%
@jit
def arrsplit(arr, ite = 2):
    res = []
    dim = arr.ndim
    a = np.array_split(arr,ite)
    for i in a: 
        b = np.array_split(i,ite,1)
        for j in b:
            if dim == 2: 
                res.append(j)
            else:
                c = np.array_split(j,ite,2) 
                for k in c:
                    res.append(k)
    return res
#%%    
def readdict(path = None, name = 'info'):
    
    '''
    readdict(path = None)
        
         path:  string, optional; 
                path where the csv-file is; reads 'info.csv'; 
                if not given, gui will open and ask for it,
                default None
    return: dict
    '''
    if not path:
        try: 
            from tkinter import Tk
            from tkinter import filedialog
            
            root = Tk()
            root.withdraw()
            
            with open(filedialog.askopenfilename(), 'rb') as f:
                r = csv.reader(f)
                return dict(r)
            
            root.destroy()
        except ImportError: 
            print('tkinter not installed, please give a path')
        
    else:
        with open(path+name+'.csv', 'rb') as f:
            r = csv.reader(f)
            return dict(r)
#%%
def figsave(path, label = None, dpi = 200, plots_path = False):
    
    if not label: label = plt.gcf().get_label()
    if plots_path: path = path+'\\Plots'
    makepath(path)
    plt.savefig(path+os.sep+label+'.png', dpi = dpi)
#%%
def npyload(path = None):
    info = {}
    
    if not path:
        from tkinter import Tk
        from tkinter import filedialog
        
        root = Tk()
        root.withdraw()
        
        with open(filedialog.askopenfilename(), 'rb') as f:
            info = {k: v for line in f for (k, v) in (line.strip().split(': ', 1),)}   
        return info
        
        root.destroy()
    else:
        with open(path+'\\info.txt') as f:
            info = {k: v for line in f for (k, v) in (line.strip().split(': ', 1),)}  
        return info
#%%
def appendSpherical_np(xyz):
    # ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    print(xyz)
    xyz = np.array(xyz)
    print(xyz)
    ptsnew = np.zeros_like(xyz)
    xy = xyz[0]**2 + xyz[1]**2
    ptsnew[0] = np.sqrt(xy + xyz[2]**2)
    ptsnew[1] = np.arctan2(np.sqrt(xy), xyz[2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[2] = np.arctan2(xyz[1], xyz[0])
    return ptsnew
#%%
def topolar(img, order=1, ret = None):
    
    from scipy.ndimage.interpolation import geometric_transform
    
    """
    Transform img to its polar coordinate representation.

    order:  int, default 1
            Specify the spline interpolation order. 
            High orders may be slow for large images.
    """
    # max_radius is the length of the diagonal 
    # from a corner to the mid-point of img.
    max_radius = 0.5*np.linalg.norm( img.shape )

    def transform(coords):
        # Put coord[1] in the interval, [-pi, pi]
        theta = 2*np.pi*coords[1] / (img.shape[1] - 1.)

        # Then map it to the interval [0, max_radius].
        #radius = float(img.shape[0]-coords[0]) / img.shape[0] * max_radius
        radius = max_radius * coords[0] / img.shape[0]

        i = 0.5*img.shape[0] - radius*np.sin(theta)
        j = radius*np.cos(theta) + 0.5*img.shape[1]
        return i,j

    polar = geometric_transform(img, transform, order=1)
    if ret:
        rads = max_radius * np.linspace(0,1,img.shape[0])
        angs = np.linspace(0, 2*np.pi, img.shape[1])
        return polar, (rads, angs)
    else:
        return polar
#%%
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    
    
    from mpl_toolkits import axes_grid1
    
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)
#%%
def fignumber():
    
    if not len(plt.get_fignums()):
        return '1'
    return str(len(plt.get_fignums())+1)
#%%
@jit
def resdouble(img, factor = 2):
    
    w,h = img.shape
    temp = np.zeros((w*factor,h*factor), dtype = img.dtype)
    for i in range(w):
        for j in range(h):
            for k in range(factor):
                for l in range(factor):
                    temp[factor*i+k,factor*j+l] = img[i,j]
    return temp
#%%
def imcolor(ar, newfig = None, cm = 'nipy_spectral', clf = True, colorbar = None, tight = True, axes = True, event = False, vmin = None, vmax = None, ret = False):
    
    '''
    imcolor(ar, newfig = True, clf = True)
        
        ar:     2D-array, either grayvalue or multichannel
        newfig: string, optional; 
                if not set a new figure will be created; 
                if given a string, it will be the name of the new figure; 
                default = True
        clf:    boolian, optional; 
                if True content of active figure will be cleared; 
                default = True
    '''
    if not cm in plt.colormaps(): 
        print('colormap {} not available, pick one of these: \n\n{}. \n\nSwitching to default value: nipy_spectral'.format(cm, plt.colormaps()))
        cm = 'nipy_spectral'
    if not newfig: newfig = fignumber()
    plt.figure(newfig,figsize = (18,8))
    if clf is True: plt.clf()
    
    im = plt.imshow(ar, cmap = cm, interpolation = 'nearest',  vmin = vmin, vmax = vmax)
    if not axes: plt.axis('off')
    if colorbar:
        add_colorbar(im)
    if tight:
        plt.tight_layout()
    if ret:
        return im
    if event == 'click': 
        fig = plt.gcf()
        fig.canvas.mpl_connect('button_press_event', onclick)
    elif event and event != 'click': 
        return plt.gcf()
#%%    
def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
#%%
def plot(ar, newfig = None, x = [], clf = True, tight = True, style = 'o-', label = '', ms = 6, event = False, **kwargs):
    
    if not newfig: newfig = fignumber()
    fig = plt.figure(newfig, figsize = (10,7))
    if clf is True: plt.clf()
    if len(x) != len(ar): x = range(len(ar))
    plt.plot(x, ar, style, ms = ms, label = label)
    if label != '': plt.legend()
    plt.tick_params(labelsize = 16)
    if kwargs:
        if 'yname' in kwargs:
            plt.ylabel(kwargs['yname'], fontsize = 16)
        if 'xname' in kwargs:
            plt.xlabel(kwargs['xname'], fontsize = 16)
        if 'title' in kwargs:
            plt.title(kwargs['title'], fontsize = 16)
    plt.grid(True)
    if event == 'click': fig.canvas.mpl_connect('button_press_event', onclick)
    elif event and event != 'click': return fig
#%%
def unitaxes(unit, axes=None, xticks = None, yticks = None, unit_name = r'$\mu m$'):
    
    if axes:
        xaxis = axes[1]
        yaxis = axes[0]
    else:
        xaxis, _, yaxis = np.abs(np.diff(plt.axis()))
    
    if not xticks:
        xticks = 10
    if not yticks:
        yticks = 10
    if isinstance(unit, str):
        unit = eval(unit)
    xtick = np.linspace(0,xaxis,xticks)
    ytick = np.linspace(0,yaxis,yticks)
    xlabel = [str(int(i*unit)) for i in xtick]
    ylabel = [str(int(i*unit)) for i in ytick]
    xlabel[-1] += unit_name
    plt.xticks(xtick, xlabel, fontsize = 16)
    plt.yticks(ytick, ylabel, fontsize = 16)
#    return [xtick, xlabel], [ytick, ylabel]
#%%
def where(arr, cond, tf=False, fillvalue=0):
    '''
    where(arr, cond, tf=False, fillvalue=0)
    
        arr:        array; 
                    array to change
        cond:       condition, mask;
        tf:         Boolian;
                    True or False, default False,
                    if True, fillvalue is replacing True values in cond
                    if False, fillvalue is replacing False values in cond
        fillvalue:  value to be filled in
    '''
    if tf: 
        return np.where(cond, fillvalue, arr)
    else:
        return np.where(cond, arr, fillvalue)
#%%
def save_obj(path, obj, name ):
    import pickle
    makepath(path)
    with open(path +os.sep+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#%%
def load_obj(path, name ):
    import pickle
    with open(path +os.sep+ name, 'rb') as f:
        return pickle.load(f)
#%%
def files(path = None):
    
    '''
    files(path = None)
        
        path:   string, optional; 
                path where the files are, if not given, gui will open and 
                ask for it,
                default None
    return: list; sorted list of the files at given path
    '''
    if not path:
        from Tkinter import Tk
        from Tkinter import filedialog
        
        root = Tk()
        root.withdraw()
        
        path = filedialog.askdirectory()
        
        onlyfiles_input = [f for f in listdir(path) if isfile(join(path, f))]
        return sorted(onlyfiles_input, key=numericalSort)
        
        root.destroy()
    else:
        onlyfiles_input = [f for f in listdir(path) if isfile(join(path, f))]
        return sorted(onlyfiles_input, key=numericalSort)
#%%
def histfix(arr, peak, ax = 0, inter = None, **kwargs):
    
    a = np.swapaxes(arr,0,ax).astype('int')
    aa = np.zeros(a.shape,dtype='int')
    count = 0
    for i in a:
        hist, bins = np.histogram(i,np.unique(i), **kwargs)
        if inter: hist = np.array(interpol(hist, inter))
        if bins[hist.argmax()] < peak:
            aa[count] = (a[count] + (peak-bins[hist.argmax()]).astype('int')).astype('int')
        elif bins[hist.argmax()] > peak:
            aa[count] = (a[count] - (bins[hist.argmax()]-peak).astype('int')).astype('int')
        else:
            aa[count] = a[count].astype('int')
                
        count +=1
    result = np.swapaxes(aa,0,ax)           
    return result
#%%
def histfix2(arr, peak, ax = 0, inter = None, **kwargs):
    
    mi = arr.min()
    ma = arr.max()
    dtype = arr.dtype
    a = np.swapaxes(arr,0,ax).astype('float')
    
    hist, bins = zip(*[np.histogram(i.flatten(),np.unique(i), **kwargs) for i in a])
    if inter: hist = np.array([interpol(hist[i], inter) for i in range(len(hist))])
    mov = np.array([bins[i][hist[i].argmax()] for i in range(len(hist))])
    a = np.array([a[i]+peak - mov[i] for i in range(len(mov))])
    result = np.swapaxes(a,0,ax)
    result = intensity2height(result, ma, mi, dtype)
    return result
#%%
def implot(ar, newfig = None, clf = True, tight = True, axes = True, vmin = None, vmax = None, ret = False):
    
#    ar = np.array(ar)
    if not ar.ndim == 2:
        raise TypeError('Dimension of Input Array must be 2')
        
    if not newfig: newfig = fignumber()
#    fig, ax = plt.subplots(1,1,num = newfig, figsize = (22,12))
    plt.figure(newfig,figsize = (22,12))
#    if clf is True: ax.cla()
    if clf: plt.clf()
    
#    ax.imshow(ar, cmap = 'gray', interpolation = 'nearest', vmin = vmin, vmax = vmax)
    plt.imshow(ar, cmap = 'gray', interpolation = 'nearest', vmin = vmin, vmax = vmax)
    if not axes: plt.axis('off')
    if tight:
        plt.tight_layout()
    if ret: return plt.gca()
#%%
def ploti(arr, name = 'orig', thresh = None, tight = True, axes = None, allinone = True):
    
    if not allinone:
        implot(arr, name)
        plt.axes('OFF')
        plt.tight_layout()
        imcolor(arr, name+'color')
        plt.axes('OFF')
        plt.tight_layout()
        if thresh:
            my_marker(arr, thresh, newfig='thresh')
            plt.axes('OFF')
            if tight:
                plt.tight_layout()
    elif allinone:
        if thresh:
            fig, ax = plt.subplots(1,3, num = name, figsize = (22,12))
            ax[0].imshow(arr, cmap = 'gray')
            ax[1].imshow(arr, cmap = 'nipy_spectral')
            ax[2].imshow(mark_boundaries(arr, arr>thresh))
            if not axes:
                for a in ax: a.set_axis_off()
#%%
def scalebar(voxelsize, unit = 'um', **kwargs):
    from matplotlib_scalebar.scalebar import ScaleBar
    scalebar = ScaleBar(voxelsize, unit, **kwargs)
    plt.gca().add_artist(scalebar)
#%%
def histstat(hist, bins):
    '''
    histstat(hist, bins)
        
            hist, bins: Data returned by "histplot" using argument "ret = True"
            returns maximum value and the standard deviation from the values to maximum value
    '''
    return bins[np.argmax(hist)], np.sqrt(sum([abs(bins[np.argmax(hist)]-i) for i in bins])*1./len(bins))
#%%
def histplot(ar, bins = None, newfig = None, clf = True, rang = None, typ = 'm', tight = True, ret = False, inter = False, density = False, **kwargs):
    
    '''
    histplot(ar, bins = None, newfig = True, clf = True, typ = 'm')
    
        ar:     N-Dim array to calculate Histogram from; Histogram is calculated from ar.flatten()
        bins:   int, Number of bins of Histogram, if none is given, bins = np.unique(ar)
        newfig: string; if not set a new figure will be created; if given a string, it will be the name of the new figure; default = True
        clf:    boolian; if True content of active figure will be cleared; default = True
        typ:    'm' or 'n'; 'm' for the box Histogram, 'n' for plot Histogram
    '''
    
    if typ != 'm' and typ != 'n': 
        raise TypeError("Wrong argument for typ; use 'm' for plt.hist, or 'n' for np.histogram")
    
    ar = np.array(ar).flatten() 
    if not rang:
        rang = [ar.min(), ar.max()]
    if bins == None:
        bins = len(np.unique(ar))
    if not newfig: newfig = fignumber()
    plt.figure(newfig, figsize = (10,7))
    if clf is True: plt.clf()
    if typ == 'm':    
        hist, bins,_  = plt.hist(ar, bins = bins, range= rang, density = density, **kwargs)
    elif typ == 'n':
        hist, bins = np.histogram(ar,bins, range=rang, density = density)
        if inter: hist = interpol(hist,inter)
        plt.plot(bins[:-1], hist, **kwargs)
    plt.grid(True)
    if tight:
        plt.tight_layout()
    if ret == True:
        return hist, bins
#%%
def ero(ar, count=1, **kwargs):
    '''
    possible kwargs: selem: neighborhood
                     out:   array to store the result
                     shift_x, shift_y: shift structuring element about center point
    '''
    ar = erosion(ar, **kwargs)
    count = count-1
    
    if count == 0:
        return ar
    else:
        return ero(ar, count)
#%%
def pearsoncor(df, name = 'pearson correlation', cmap = plt.cm.Reds, **kwargs):
    import seaborn as sns
    #Using Pearson Correlation
    plt.figure(name, figsize=(22,12))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=cmap, **kwargs)
#%%
def threshfinder(arr, vpercent):

    jo = arr.min()
    je = arr.max()
    factor = 100./arr.size
    j = jo+(je-jo)/2
    eps = 1e-4
    var = 0.
    while not np.allclose(var, vpercent, atol = eps): 
        var = np.count_nonzero(arr > j)*factor
#        print(var, j)
#        print(jo,je)
        
        if np.allclose(je,jo, atol = 1): 
            return j

        if var > vpercent:
            jo = j
            j += (je-j)/2
        elif var < vpercent:
            je = j
            j -= (j - jo)/2
    return j
#%%
def makepath(path):
    if not os.path.isdir(path):
        os.makedirs(path)
#%%
def plot_dir(path):
    date = time.strftime('%d%m%Y')
    # zeit = time.strftime('%H_%M')
    myrealpath = os.path.join(path,date)+'\\'

    if not os.path.isdir(myrealpath):
        os.makedirs(myrealpath)
        
    return myrealpath  
#%%
def random_color(outformat = 'i'):
    
    '''
    random_color(outformat = 'i')
    
        outformat:  variable type for random number, 
                    'i' for int, 'f' for float;
                    default = 'i'
        
        returns:    tuple of 3 random int in range [0:255] for 'i' or
                    tuple of 3 random floats in range [0:1] for 'f'
    '''
    color = np.random.randint(0, 255, size = 3)
    if outformat == 'f': color = color/255.
    return tuple(color)
#%%
def my_marker(arr,thresh,newfig = None, clf = True, percentlist = None, save = None, axetext = 'Threshold value', reti = None, ticks = 5):
    
#    arr = np.where(arr<1,0,arr)
    dat1 = arr.astype('float32')
    if isinstance(thresh,list):
        val = [x/dat1.max() for x in map(float,thresh)]
        if len(thresh) == 3:
            import matplotlib
#            colorsList = [(169./255,169./255,169./255), (0,0,0.8), (0.8,0,0)]
            colorsList = [(0,0,0), (0,0,0.8), (0.8,0,0)]
            CustomCmap = matplotlib.colors.ListedColormap(colorsList)
            matplotlib.cm.register_cmap(cmap = CustomCmap)
            cmap = plt.get_cmap(CustomCmap, len(thresh))
        else:
            cmap = plt.get_cmap('jet',len(thresh))
#        cmap.set_under('gray')
#        bounds = np.linspace(0,len(thresh),len(thresh)+1)
        bounds = np.linspace(0,len(thresh),ticks+1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    else:
        val = float(thresh)/dat1.max()
    dat1 = dat1/dat1.max()
    w, h = dat1.shape
    ret = np.empty((w, h, 3), dtype=np.float32)
    ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  dat1.astype('float32')
    
    if not isinstance(thresh, list):
        ret[dat1>=val] = (0.4,0,0)
    else: 
        for i in val:
            ret[dat1>=i] = cmap(val.index(i))[:3]
    if not newfig: newfig = fignumber()   
    fig =plt.figure(newfig, figsize = (10,7))
    if clf == True: fig.clf()

    ax = fig.add_subplot(111)
    
    if isinstance(thresh,list):
        cax = ax.imshow(ret,cmap = cmap, norm = norm)
        cbar = add_colorbar(cax)
        cbar.set_norm = norm
        cbar.set_spacing = 'proportional'
#        cbar = fig.colorbar(cax,norm = norm, spacing = 'proportional')#, extend = 'min')#, boundaries = bounds)
        threshticks = np.linspace(min(thresh), arr.max(), 11)
#        cbar.ax.set_yticklabels(thresh+[arr.max()],fontsize = 16)
        cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(11))
        if ticks == len(thresh) and len(thresh) <= 10: 
            cbar.ax.set_yticklabels(np.array(thresh+[arr.max()]).astype(int),fontsize = 16)
        else:
            cbar.ax.set_yticklabels(threshticks.astype(int),fontsize = 16)
        if isinstance(percentlist,list):
            for j, lab in enumerate(percentlist):
                cbar.ax.text(0.5,0.11+j*1.0/len(percentlist), lab, ha = 'center', va = 'center', color = 'white', fontsize = 16, rotation = 270, weight = 'bold')
#        cbar.set_label('Threshold value',rotation = 360, position = [0.5, 0.5, 0.3, 0.5], fontsize = 12)
#        cbar.ax.text(1.5,0.55,axetext, rotation = 270, fontsize = 26)
        plt.tight_layout()
        if reti: 
            return cax
    else:
        cax = ax.imshow(ret)
        plt.tight_layout()
        if reti: 
            return cax
    if save:
        
        plt.savefig(save)
#%%
def wavelett(arr, nplot = 'all',axis = 'ON'):
    
    import pywt

    wp = pywt.WaveletPacket2D(data=arr, wavelet = 'db1')
    
    typ = ['a', 'd', 'h', 'v']
    if nplot == 'all':
        for i in typ:
            implot(wp[i].data, i)
#        plt.tight_layout()
    
    fig = plt.figure('wavelett', figsize=(10,7))
    plt.clf()
#    fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(nrows = 2, ncols = 2)
    
    for i, j in enumerate(typ):
        ax = fig.add_subplot(221+i)
        ax.imshow(wp[j].data, cmap = 'gray')
        ax.set_title(j)
        ax.axis(axis)
    fig.tight_layout()
#%%
def torgb(arr):
    arr = arr[...,None]
    return np.concatenate((arr,arr,arr), axis = -1)
#%%
def colorfloat(arr):
    
    img = arr*1./arr.max()
    img = img[...,None].astype(float)
    ret2 = np.concatenate((img,img,img), axis = -1)
    
    return img_as_float(ret2)   
#%%
def autothresh(arr, bins = None, order = 20, newhist = None, newfig = None, ret = False, pol = False, ticks = 5, **kwargs):
    
    arr = np.array(arr)

    if bins == None:
        bins = len(np.unique(arr))

    plt.figure(newhist, figsize = (10,7))
    plt.clf()

    hist, bins = np.histogram(arr.flatten(),bins, **kwargs)
    if pol:
        hist = np.array(interpol(hist, pol))
    amin = argrelextrema(hist, np.less, order = order)
    plt.plot(bins[:-1], hist)
    plt.plot(bins[amin[0]], hist[amin[0]], 'ro')
    plt.grid()
    plt.tight_layout()
    
    thresh = [arr.min()]+bins[amin[0]].tolist()
    
    if not newfig: newfig = fignumber()
    my_marker(arr, thresh = thresh, newfig = newfig, ticks = ticks) 
    
    if ret == True:
        return thresh
#%%
    
        
        