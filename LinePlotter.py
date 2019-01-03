import matplotlib as mpl
mpl.use('Agg')
import numpy
import scipy.ndimage
from astropy.stats import sigma_clip
# from matplotlib import rcParams
# rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
from astropy.coordinates import ICRS
from astropy import units as u
from astropy import wcs
from astropy.io import fits
import os,sys
from scipy import stats
from collections import Counter
import seaborn as sns
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from astropy.modeling.models import Gaussian2D
from sklearn.cluster import DBSCAN
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from matplotlib.patches import Rectangle
from astropy.io import ascii
from astropy.cosmology import Planck13 as cosmology
import itertools
import numpy as np
import aplpy
import os.path
import emcee
sns.set_style("white", {'legend.frameon': True})
sns.set_style("ticks", {'legend.frameon': True})
sns.set_context("talk")
sns.set_palette('Dark2',desat=1)
cc = sns.color_palette()
import yaml
import argparse


def lnprior(theta):
    amp, b, c = theta
    if min_flux < amp < max_flux and minf < b < maxf and 0 < c < 0.5:
        return 0.0
    return -np.inf


def lnlike(theta, measured_flux,sigma,x):
    y_model = theta[0] * np.exp(-0.5 * (x - theta[1])**2 / theta[2]**2)
    return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (measured_flux - y_model) ** 2 / sigma ** 2)

def lnprob(theta, measured_flux,sigma,x):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, measured_flux,sigma,x)

def get_beam(file):
  try:
    head = fits.open(file)[0].header
    #Jy/beam to Jy/pix
    bmaj = head['BMAJ']*3600.0
    bmin = head['BMIN']*3600.0
    bpa = head['BPA']
    pix_size = head['CDELT2']*3600.0
  except:
    head = fits.open(file)[0].header
    table = fits.open(file)[1].data
    #Jy/beam to Jy/pix
    bmaj = table['BMAJ'][0]
    bmin = table['BMIN'][0]
    bpa = table['BPA'][0]
    pix_size = head['CDELT2']*3600.0
  factor = 2*(numpy.pi*bmaj*bmin/(8.0*numpy.log(2)))/(pix_size**2)
  factor = 1.0/factor
  return bmaj,bmin,factor,bpa



def fill_between_steps(ax, x, y1, y2=0, step_where='mid', **kwargs):
    ''' fill between a step plot and 

    Parameters
    ----------
    ax : Axes
       The axes to draw to

    x : array-like
        Array/vector of index values.

    y1 : array-like or float
        Array/vector of values to be filled under.
    y2 : array-Like or float, optional
        Array/vector or bottom values for filled area. Default is 0.

    step_where : {'pre', 'post', 'mid'}
        where the step happens, same meanings as for `step`

    **kwargs will be passed to the matplotlib fill_between() function.

    Returns
    -------
    ret : PolyCollection
       The added artist

    '''
    if step_where not in {'pre', 'post', 'mid'}:
        raise ValueError("where must be one of {{'pre', 'post', 'mid'}} "
                         "You passed in {wh}".format(wh=step_where))

    # make sure y values are up-converted to arrays 
    if np.isscalar(y1):
        y1 = np.ones_like(x) * y1

    if np.isscalar(y2):
        y2 = np.ones_like(x) * y2

    # temporary array for up-converting the values to step corners
    # 3 x 2N - 1 array 

    vertices = np.vstack((x, y1, y2))

    # this logic is lifted from lines.py
    # this should probably be centralized someplace
    if step_where == 'pre':
        steps = np.zeros((3, 2 * len(x) - 1), np.float)
        steps[0, 0::2], steps[0, 1::2] = vertices[0, :], vertices[0, :-1]
        steps[1:, 0::2], steps[1:, 1:-1:2] = vertices[1:, :], vertices[1:, 1:]

    elif step_where == 'post':
        steps = np.zeros((3, 2 * len(x) - 1), np.float)
        steps[0, ::2], steps[0, 1:-1:2] = vertices[0, :], vertices[0, 1:]
        steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :-1]

    elif step_where == 'mid':
        steps = np.zeros((3, 2 * len(x)), np.float)
        steps[0, 1:-1:2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
        steps[0, 2::2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
        steps[0, 0] = vertices[0, 0]
        steps[0, -1] = vertices[0, -1]
        steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :]
    else:
        raise RuntimeError("should never hit end of if-elif block for validated input")

    # un-pack
    xx, yy1, yy2 = steps

    # now to the plotting part:
    return ax.fill_between(xx, yy1, y2=yy2, **kwargs)

def GetSN(data,i):
  data2 = data[i]
  data2 = np.mean(data2,axis=0)
  noise = numpy.std(data2[data2!=0.0])
  if np.isnan(data2[pix_y][pix_x]/noise):
    return 0
  else:
    return data2[pix_y][pix_x]/noise

def GetSNFast(Spaxel,rms,i):
  return np.sum(Spaxel[i])/np.sqrt(np.sum(np.power(rms[i],2)))

def GetSigma(data,i):
  data2 = data[i]
  data2 = np.mean(data2,axis=0)
  noise = numpy.std(data2[data2!=0.0])
  return noise

def GetRMSarray(data):
  rms = []
  for i in range(len(data)):
    rms.append(np.nanstd(data[i][data[i]!=0.0]))
  return np.array(rms)



def PlotSPW(ff,x,y,data):
  # print 'plotting:',ff
  hdulist =   fits.open(ff,memmap=True)
  w   =   WCS(hdulist[0].header)
  header  =   hdulist[0].header
  XX = []
  YY = []
  rms = []
  for i in range(len(data)):
    XX.append(i)
    YY.append(data[i][y][x])
    rms.append(np.std(data[i][data[i]!=0.0]))
  aux = numpy.array([x*numpy.ones(len(XX)),y*numpy.ones(len(XX)),XX,numpy.zeros(len(XX))])
  aux = numpy.transpose(aux)
  aux = w.all_pix2world(aux,0)
  aux = numpy.transpose(aux)
  XX = aux[2]/1e9
  YY = numpy.array(YY)*1000.0
  rms = np.array(rms)*1000.0
  rms[np.isnan(rms)] = 1.0
  return XX,YY,aux,rms

def PlotSPW2(ff,x,y,data,factor):
  # print 'plotting:',ff
  hdulist =   fits.open(ff,memmap=True)
  w   =   WCS(hdulist[0].header)
  header  =   hdulist[0].header
  XX = []
  YY = []
  rms = []
  for i in range(len(data)):
    XX.append(i)
    YY.append(sum(data[i][y,x]))
    rms.append(np.std(data[i][data[i]!=0.0]))
  aux = numpy.array([x[0]*numpy.ones(len(XX)),y[0]*numpy.ones(len(XX)),XX,numpy.zeros(len(XX))])
  aux = numpy.transpose(aux)
  aux = w.all_pix2world(aux,0)
  aux = numpy.transpose(aux)
  XX = aux[2]/1e9
  YY = numpy.array(YY)*1000.0*factor
  rms = np.array(rms)*1000.0*np.sqrt(len(y))*np.sqrt(factor)
  # print len(y),1.0/factor
  rms[np.isnan(rms)] = 1.0
  return XX,YY,aux,rms



def prune(samples,lnprob, scaler=5.0, quiet=True):

    minlnprob = lnprob.max()
    dlnprob = numpy.abs(lnprob - minlnprob)
    medlnprob = numpy.median(dlnprob)
    avglnprob = numpy.mean(dlnprob)
    skewlnprob = numpy.abs(avglnprob - medlnprob)
    rmslnprob = numpy.std(dlnprob)
    inliers = (dlnprob < scaler*rmslnprob)
    lnprob2 = lnprob[inliers]
    samples = samples[inliers]

    medlnprob_previous = 0.
    while skewlnprob > 0.1*medlnprob:
        minlnprob = lnprob2.max()
        dlnprob = numpy.abs(lnprob2 - minlnprob)
        rmslnprob = numpy.std(dlnprob)
        inliers = (dlnprob < scaler*rmslnprob)
        PDFdatatmp = lnprob2[inliers]
        if len(PDFdatatmp) == len(lnprob2):
            inliers = (dlnprob < scaler/2.*rmslnprob)
        lnprob2 = lnprob2[inliers]
        samples = samples[inliers]
        dlnprob = numpy.abs(lnprob2 - minlnprob)
        medlnprob = numpy.median(dlnprob)
        avglnprob = numpy.mean(dlnprob)
        skewlnprob = numpy.abs(avglnprob - medlnprob)
        if not quiet:
            print(medlnprob, avglnprob, skewlnprob)
        if medlnprob == medlnprob_previous:
            scaler /= 1.5
        medlnprob_previous = medlnprob
    samples = samples[lnprob2 <= minlnprob]
    lnprob2 = lnprob2[lnprob2 <= minlnprob]
    return samples,lnprob2

def GetCollapsedLineProperties(x,y,rms,i_final):
  FirstChannel = i_final[0]
  LastChannel = i_final[-1]

  while y[FirstChannel]>0 and FirstChannel>0:
    FirstChannel = FirstChannel - 1
  FirstChannel = FirstChannel + 1

  while y[LastChannel]>0 and LastChannel<(len(y)-1):
    LastChannel = LastChannel + 1
  LastChannel = LastChannel 

  dv = (x[1]-x[0])*299792.458/x[1]
  IntegratedFlux = np.sum(y[FirstChannel:LastChannel]*dv)
  ErrorIntegratedFlux = np.sqrt(np.sum(np.power(rms[FirstChannel:LastChannel]*dv,2))) 

  i_final = np.arange(FirstChannel,LastChannel)
  a = []
  for ll in range(1000):
    newy = np.random.normal(y[i_final],rms[i_final])
    a.append(np.sum(x[i_final]*newy)/np.sum(newy))
  aux = np.percentile(a,[16,50,84])
  Moment = aux[1]
  ErrorMoment = np.mean([aux[1]-aux[0],aux[2]-aux[1]])
  return FirstChannel,LastChannel,Moment,ErrorMoment,IntegratedFlux,ErrorIntegratedFlux



def CreateConfigFile():
    cmd = "CatalogLines: LineCandidatesPositive.dat\nCubePath: /data2/aspecs/band6/CollapsedCubeBand6/SmoothCube_contsub.fits\nPBpath: /data2/aspecs/band6/UDF_mosaic_1mm.cube.60mhz.pb.fits\nOutputFolder: OutputLinePlot\nLimitInSN: True\nLimitInP: True\nSNLimit: 5.4\nPLimit: 0.1\nColorFitsFile: /data2/aspecs/band6/Cube3/XDF_cube_2d.fits\nColorImage: /data2/aspecs/band6/Cube3/XDF_rgb.png\nSurveyName: ASPECS-LP-1mm\nSaveCollapsedLines: False\nValueAddEachSide: 20\nNwalkers: 100\nNsteps: 100\nParameterForMCMC: 3"
    output = open('configLinePlot.yaml','w')
    output.write(cmd)
    output.close()
##################################################################################################################
##################################################################################################################
##################################################################################################################

#Parse the input arguments
parser = argparse.ArgumentParser(description="Python script that plots emission lines candidates")
parser.add_argument('--CreateConfigFile', action='store_true',required=False,help = 'Create template configuration file')


args = parser.parse_args()
#Checking input arguments
print 20*'#','Checking inputs....',20*'#'
if args.CreateConfigFile:
    CreateConfigFile()
    print '*** Creating configuration file ***'

ConfigFile = yaml.safe_load(open('configLinePlot.yaml'))


CatalogLines = ConfigFile['CatalogLines']
CubePath = ConfigFile['CubePath']
PBpath = ConfigFile['PBpath']
OutputFolder = ConfigFile['OutputFolder']
candidates = ascii.read(CatalogLines)

if os.path.isdir(OutputFolder):
	os.system('rm -rf '+OutputFolder+'/*')
	os.mkdir(OutputFolder+'/plots')
else:
	os.mkdir(OutputFolder)
	os.mkdir(OutputFolder+'/plots')

if ConfigFile['LimitInSN']:
	candidates = candidates[candidates['SN']>=ConfigFile['SNLimit']]

if ConfigFile['LimitInP']:
	candidates = candidates[candidates['PPoisExp']<=ConfigFile['PLimit']]

p = 1.0 - candidates['PPoisExp']
pE1 = candidates['PPoisExpE2']
pE2 = candidates['PPoisExpE1']
print 'Expected to be real:',np.sum(p)*1.0,'-',np.sqrt(np.sum(np.power(pE1,2))),'+',np.sqrt(np.sum(np.power(pE2,2)))
print 50*'#'

freq = candidates['FREQ']
ra = candidates['RA']
dec = candidates['DEC']
hdulist =   fits.open(CubePath,memmap=True)
PBFile = fits.open(PBpath,memmap=True)
head = hdulist[0].header


try:
    BMAJ = hdulist[1].data.field('BMAJ')
    BMIN = hdulist[1].data.field('BMIN')
    BPA = hdulist[1].data.field('BPA')

except:
    BMAJ = []
    BMIN = []
    BPA = []
    for i in range(int(head['NAXIS3'])):
        BMAJ.append(head['BMAJ']*3600.0)
        BMIN.append(head['BMIN']*3600.0)
        BPA.append(head['BPA'])
    BMAJ = np.array(BMAJ)
    BMIN = np.array(BMIN)
    BPA = np.array(BPA)

pix_size = head['CDELT2']*3600.0
factor = 2*(numpy.pi*BMAJ*BMIN/(8.0*numpy.log(2)))/(pix_size**2)
factor = 1.0/factor
w = wcs.WCS(hdulist[0].header)
[lala,lala,FreqReference,lala] =  w.all_pix2world(np.zeros_like(range(head['NAXIS3'])),np.zeros_like(range(head['NAXIS3'])),range(head['NAXIS3']),np.zeros_like(range(head['NAXIS3'])),0)
c = []
radeg = []
decdeg = []
for i in range(len(ra)):
  c.append(SkyCoord(ra[i], dec[i], frame='icrs', unit=(u.hourangle, u.deg)))
  radeg.append(SkyCoord(ra[i], dec[i], frame='icrs', unit=(u.hourangle, u.deg)).ra.deg)
  decdeg.append(SkyCoord(ra[i], dec[i], frame='icrs', unit=(u.hourangle, u.deg)).dec.deg)

[xxx2,yyy2,channel,lala] = w.all_world2pix(radeg,decdeg,np.zeros_like(radeg),freq*1e9,0,ra_dec_order=True)
channel = np.interp(freq*1e9,FreqReference,range(head['NAXIS3']))
for i in range(len(xxx2)):
  k = i + 1
  print 'ASPECS-LP-1mm.'+str(k).zfill(2)+' & '+c[i].to_string('hmsdms',sep=':',precision=2).split()[0]+' & '+c[i].to_string('hmsdms',sep=':',precision=2).split()[1]+' & '+str(candidates['FREQ'][i])+' & '+str(candidates['SN'][i])+' & $'+str(round(1-candidates['PPoisExp'][i],2))+'_{-'+str(round(candidates['PPoisExpE2'][i],2))+'}^{+'+str(round(candidates['PPoisExpE1'][i],2))+'}$\\\\'

fitted_gaussian = open(OutputFolder+'/gaussian_lines_fit_values.dat','w')
fil = CubePath
print 50*'#'
print 'reading file...'
data = fits.open(fil)[0].data[0]
data = 1.0*np.nan_to_num(data)
NumberFinalSample = 1
isPS = open(OutputFolder+'/isPS.txt','w')
PBvalues = PBFile[0].data[0,channel.astype(int),yyy2.astype(int),xxx2.astype(int)]

MasterRMS = GetRMSarray(data)
for j in range(len(xxx2)):
  k = j+1
  EL_name = ConfigFile['SurveyName']+'.'+str(NumberFinalSample).zfill(2)
  pix_x = int(xxx2[j])
  pix_y = int(yyy2[j])
  xr = pix_x
  yr = pix_y
  output = OutputFolder+'/plots/'+ConfigFile['SurveyName']+'_'+str(NumberFinalSample).zfill(2)+'.pdf'
  NumberFinalSample += 1

  print 50*'#'
  print fil,xr,yr,channel[j]

  # x = []
  # y = []
  # for i in range(len(data)):
  #   y.append(data[i][pix_y][pix_x])
  #   x.append(i)
  # x = np.array(x)
  # y = np.array(y)
  # print np.shape(data)
  x = np.arange(len(data))
  Spaxel =  np.transpose(data)[pix_x,pix_y]
  initial_channel = int(channel[j])


  # sn_initial = np.array([GetSN(data,np.array([int(initial_channel-1),int(initial_channel),int(initial_channel+1)])),])
  sn_initial = np.array([GetSNFast(Spaxel,MasterRMS,np.array([int(initial_channel-1),int(initial_channel),int(initial_channel+1)])),])
  print 'MaxSN:',round(sn_initial[0],1),'Channel:',int(initial_channel-1)
  if sn_initial == 0:
  	line2text = EL_name + ' & -1 \pm -1 & -1 \pm -1 & -1 \pm -1 \\\\\n'
  	print line2text
  	fitted_gaussian.write(line2text)
  	tmp = EL_name+' -1 -1 NL -1 -1\n'
  	isPS.write(tmp)
  	fitted_gaussian.flush()
  	isPS.flush()
  	continue
  # print 'max sn:',max(sn_initial),channel[j],np.argmax(sn_initial)
  index = []
  sn = []
  contador = 0
  j = 0
  br = False
  ValueAddEachSide = ConfigFile['ValueAddEachSide']
  while contador<10:
    if initial_channel!=0:
      i = np.arange(np.random.randint(low=max(0,initial_channel-ValueAddEachSide),high=initial_channel),np.random.randint(low=initial_channel+1,high=min(len(x)+1,initial_channel+ValueAddEachSide)),1)
    else:
      i = np.arange(0,np.random.randint(low=initial_channel+1,high=min(len(x)+1,initial_channel+ValueAddEachSide)),1)

    index.append(i)
    # aux = GetSN(data,i)
    aux = GetSNFast(Spaxel,MasterRMS,i)
    if aux>=0.9*max(sn_initial):
      contador+=1
    print 'First iteration:',contador,j,i[0],i[-1],round(aux,2)
    j += 1
    if j>=1000 and contador==0:
      br = True
      break
    if j==250 or j==500 or j==1000  or j==2000 or j==3000:
      ValueAddEachSide = ValueAddEachSide/2
    sn.append(aux)

  if br:
    continue
  l = []
  for i in index:
    l.append(len(i))
  l = np.array(l)
  sn = np.array(sn)
  max_l = int(max(l[sn>=0.9*max(sn_initial)])*0.5)
  index = []
  sn = []
  j = 0
  for i in itertools.combinations(np.arange(max(initial_channel-max_l-1,0),min(len(x),initial_channel+1+max_l+1)),2):
    i = np.arange(i[0],i[1])
    index.append(i)
    # aux = GetSN(data,i)
    aux = GetSNFast(Spaxel,MasterRMS,i)
    print 'Second iteration:',j,i[0],i[-1],round(aux,2)
    sn.append(aux)
    j +=1

  l = []
  for i in index:
    l.append(len(i))
  l = np.array(l)
  sn = np.array(sn)
  index = np.array(index)
  i = index[np.argmax(sn)]
  i_final = index[np.argmax(sn)]
  # plt.plot(x,y,drawstyle='steps-mid')
  # plt.title('SN:'+str(round(GetSN(data,i),1)))
  idi = i[0]
  idf = i[-1]

  print 'Channels:',i,
  print 'SN:',round(GetSN(data,i),1)
  max_sn_yellow_contour = GetSN(data,i)


  sigma = GetSigma(data,i_final)
  data2 = np.mean(data[i_final],axis=0)
  SNForPlot = data2/sigma
  X = np.array(np.where(SNForPlot>=2.0))
  X = np.transpose(X)
  db = DBSCAN(eps=3, min_samples=1).fit(X)
  core_samples_mask = numpy.zeros_like(db.labels_, dtype=bool)
  core_samples_mask[db.core_sample_indices_] = True
  labels = db.labels_
  n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
  unique_labels = set(labels)
  colors = plt.cm.Spectral(numpy.linspace(0, 1, len(unique_labels)))

  XR = []
  YR = []
  flux_point_source = []
  for k, col in zip(unique_labels, colors):
    if k == -1:
      col = 'k'
    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    if xr in xy[:, 1] and yr in xy[:, 0]:
      XR = xy[:, 1]
      YR = xy[:, 0]

  w, h = 1.0*plt.figaspect(0.5)
  fig = plt.figure(figsize=(w,h))
  plt.subplots_adjust(left=0.14, bottom=0.17, right=0.97, top=0.94,wspace=0.10, hspace=0.0)
  ax1 = plt.subplot(111)

  if len(XR)==0:
  	line2text = EL_name + ' & 0 \pm 0 & 0 \pm 0 & 0 \pm 0 \\\\\n'
  	print line2text
  	fitted_gaussian.write(line2text)
  	tmp = EL_name+' 0 0 NL 0 0\n'
  	isPS.write(tmp)
  	fitted_gaussian.flush()
  	isPS.flush()
  	continue
  isPS.write(EL_name+' '+str(len(XR))+' ' + str(1.0/factor[initial_channel])+' ')

  XXPS,YYPS,auxPS,rmsPS = PlotSPW(fil,xr,yr,data)
  XXEXT,YYEXT,auxEXT,rmsEXT = PlotSPW2(fil,XR,YR,data,factor[initial_channel])
  if sum(YYEXT[i_final])<1.1*sum(YYPS[i_final]):
    XX = XXPS
    YY = YYPS
    aux = auxPS
    rms = rmsPS
    isPS.write('PS '+str(sum(YYPS[i_final]))+' '+str(sum(YYEXT[i_final]))+'\n')
    # print 'PS',sum(YYPS[i_final]),sum(YYEXT[i_final])
  else:
    XX = XXEXT
    YY = YYEXT
    aux = auxEXT
    rms = rmsEXT  
    isPS.write('EXT '+str(sum(YYPS[i_final]))+' '+str(sum(YYEXT[i_final]))+'\n')
    # print 'EXT',sum(YYPS[i_final]),sum(YYEXT[i_final])
  isPS.flush()
  x,y = XX,YY
  
  FirstChannel,LastChannel,Moment,ErrorMoment,IntegratedFlux,ErrorIntegratedFlux = GetCollapsedLineProperties(x,y/PBvalues[NumberFinalSample-2],rms/PBvalues[NumberFinalSample-2],i_final)
  FrequencyInitialFlux = x[FirstChannel]
  FrequencyFinalFlux = x[LastChannel-1]

  if idi!=0 and idf!=len(x)-1:
    xc = np.append(x[idi-1],x[i])
    xc = np.append(xc,x[idf+1])
    yc = np.append(0,y[i])
    yc = np.append(yc,0)
  elif idi==0 and idf!=len(x)-1:
    xc = x[i]
    yc = y[i]
    xc = np.append(xc,x[idf+1])
    yc = np.append(yc,0)
  elif idi!=0 and idf==len(x)-1:
    xc = np.append(x[idi-1],x[i])
    yc = np.append(0,y[i])
  else:
    xc = x[i]
    yc = y[i]
  fill_between_steps(ax1,xc,yc/PBvalues[NumberFinalSample-2],color='yellow',zorder=1)

  lim_negative = XX[initial_channel]-1
  lim_positive = XX[initial_channel]+1
  YY = YY[XX>=lim_negative]/PBvalues[NumberFinalSample-2]
  rms = rms[XX>=lim_negative]/PBvalues[NumberFinalSample-2]
  XX = XX[XX>=lim_negative]

  YY1 = YY[XX<=lim_positive]
  rms1 = rms[XX<=lim_positive]
  XX1 = XX[XX<=lim_positive]

  x,y = XX1,YY1
  axx = ax1

  ax1.axhline(0,color='black',lw=1)
  ax1.plot(XX1[YY1!=0],YY1[YY1!=0],drawstyle='steps-mid',linewidth=1.5,color=sns.color_palette()[0],zorder=2)
  ax1.errorbar(XX1[YY1!=0],YY1[YY1!=0],yerr=rms1[YY1!=0],fmt=',',ls='',color='gray',lw=1)
  x = 1.0*XX1
  minf = min(x)
  maxf = max(x)
  measured_flux = YY1
  sigma = rms1
  # print 'frequency:',x[np.argmax(measured_flux)]
  # minf = max(x[np.argmax(measured_flux)] - 0.5,min(x))
  # maxf = min(x[np.argmax(measured_flux)] + 0.5,max(x))
  minf = max(np.mean(x) - 0.5,min(x))
  maxf = min(np.mean(x) + 0.5,max(x))

  min_flux = max(measured_flux)*0.1
  max_flux = max(measured_flux)*5.0
  nwalkers = ConfigFile['Nwalkers']  # number of MCMC walkers
  nsteps = ConfigFile['Nsteps']  # number of MCMC steps to take
  nburn = int(0.75*nsteps)  # "burn-in" period to let chains stabilize
  ndim = 3
  # initial_estimate = [max(measured_flux),x[np.argmax(measured_flux)],0.01]
  initial_estimate = [max(measured_flux),np.mean(x),0.01]
  starting_guesses = []
  for i in range(nwalkers):
      aux2 = []
      aux2 = [np.random.uniform(0.0,max(measured_flux)*5.0),initial_estimate[1]+ np.random.uniform(-0.1,0.1),np.random.uniform(0.001,0.2)]
      starting_guesses.append(np.array(aux2))
  starting_guesses = np.array(starting_guesses)   
  print 70*'-'

  print 'Number of iterations:',ndim*nwalkers*nsteps
  sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[measured_flux,sigma,x], threads=1,a=ConfigFile['ParameterForMCMC'])
  sampler.run_mcmc(starting_guesses, nsteps)
  af = sampler.acceptance_fraction
  print "Mean acceptance fraction:", np.mean(af)

  af_msg = '''As a rule of thumb, the acceptance fraction (af) should be 
                              between 0.2 and 0.5
              If af < 0.2 decrease the a parameter
              If af > 0.5 increase the a parameter
              '''
  if np.mean(af)<0.2 or np.mean(af)>0.5:
    print af_msg
  emcee_trace = sampler.chain[:, :, :].reshape((-1, ndim))
  lnprob_aux = sampler.lnprobability
  # print 'best fit:',emcee_trace[np.argmax(lnprob_aux)]
  theta = emcee_trace[np.argmax(lnprob_aux)]
  samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
  lnprob_aux = sampler.lnprobability[:, nburn:].reshape(-1)
  lnprob_flat = sampler.flatlnprobability[int(0.75*nsteps*nwalkers):]
  # print 'shape lnprob_aux:',np.shape(lnprob_aux)
  # print 'shape samples:',np.shape(samples)
  ll = ['Amp','f0','Sigma']
  samples,lnprob2 = prune(samples,lnprob_aux)
  values = []
  values_errors = []
  for ID in range(3):
      pc = np.percentile(samples.T[ID], [16,50,84])
      print ll[ID]+':',round(pc[1],4),'+/-',round(np.mean([pc[2]-pc[1],pc[1]-pc[0]]),4),pc
      values.append(pc[1])
      values_errors.append(np.mean([pc[2]-pc[1],pc[1]-pc[0]]))


  Velocity = int(round(values[2]*2.3548*299792.458/values[1],0))
  VelocityError = int(round(values_errors[2]*2.3548*299792.458/values[1],0))
  ax1.text(0.02, 0.95,'Amplitude: '+str(round(values[0],2))+' +/- '+str(round(values_errors[0],2))+r' mJy beam$^{-1}$'+'\nFrequency: '+str(round(values[1],3))+' +/- '+str(round(values_errors[1],3))+' GHz\nFWHM: '+str(Velocity)+' +/- '+str(VelocityError)+r' km s$^{-1}$', horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes,fontsize=18,bbox={'facecolor':'white', 'alpha':1.0, 'pad':2})
  ax1.text(0.96, 0.95,EL_name, horizontalalignment='right',verticalalignment='top',transform=ax1.transAxes,fontsize=18,bbox={'facecolor':'white', 'alpha':1.0, 'pad':2})

  line2text = EL_name + ' & '+str(round(values[0],2))+' \pm '+str(round(values_errors[0],2)) + ' & '+str(round(values[1],3))+' \pm '+str(round(values_errors[1],3))+' & '+str(round(values[2]*2.3548*299792.458/values[1],0))+' \pm '+str(round(values_errors[2]*2.3548*299792.458/values[1],0))+' '+ str(round(Moment,3))+' '+str(round(ErrorMoment,3))+' '+str(round(IntegratedFlux,0))+' '+str(round(ErrorIntegratedFlux,0)) +'\\\\\n'
  
  print line2text
  fitted_gaussian.write(line2text)
  fitted_gaussian.flush()

  xx = np.linspace(min(x),max(x),1000)
  yy = values[0] * np.exp(-0.5 * (xx - values[1])**2 / values[2]**2)
  ax1.plot(xx,yy,lw=1.5,color=sns.color_palette()[1],zorder=3)
  ax1.axhline(0,color='black',lw=0.5,zorder=0)
  ax1.axvline(FrequencyInitialFlux,color='black',ls='--',lw=1,zorder=0)
  ax1.axvline(FrequencyFinalFlux,color='black',ls='--',lw=1,zorder=0)
  ax = plt.gca()
  plt.tick_params(axis='both', which='major', labelsize=20)
  ax1.set_xlabel(r'$\nu$ [GHz]',fontsize=20)
  ax1.set_ylabel(r'$F_{\nu}$ [mJy b$^{-1}$]',fontsize=20)
  ax1.set_xlim(min(xx)-0.05,max(xx)+0.05)
  ax1.set_ylim(min(measured_flux-sigma)-0.2,2.0*max(measured_flux+sigma))
  plt.savefig(output)
  plt.close()

  index = index[sn>=max(sn_initial)]

  l = l[sn>=max(sn_initial)]
  i = index[np.argmax(l)]

  idi = i[0]
  idf = i[-1]

  w = WCS(fil)
  w2 = WCS(fil)

  aux = w.all_pix2world([[pix_x,pix_y,0,0]],1)
  # print aux
  ra2 = aux[0][0]
  dec2 = aux[0][1]
  hdulist_aux = fits.open(ConfigFile['ColorFitsFile'])

  # print 'Beam:',BMAJ[initial_channel],BMIN[initial_channel]
  hdulist_aux[0].header.set('BMAJ',BMAJ[initial_channel]/3600.0)
  hdulist_aux[0].header.set('BMIN',BMIN[initial_channel]/3600.0)
  hdulist_aux[0].header.set('BPA',BPA[initial_channel])
  os.system('rm -rf 2d_temp.fits')
  hdulist_aux.writeto('2d_temp.fits',clobber=True)
  f = aplpy.FITSFigure('2d_temp.fits',subplot=[0.17,0.1,0.75,0.85])
  w2 = WCS('2d_temp.fits')
  f.show_rgb(ConfigFile['ColorImage'])
  sigma = GetSigma(data,i_final)
  # print 'Sigma:',sigma
  levels = numpy.array([-5,-4,-3,3,4,5,6,7,8,9,10])
  lws = numpy.ones(len(levels))*1.5
  lws[levels<0]=0.5
  levels = sigma*levels
  f.add_beam()
  f.beam.show(frame=True,facecolor='red',edgecolor='black',hatch='/')
  data2 = data[i_final]
  data2 = np.mean(data2,axis=0)
  hdulist[0].data = [[data2]]
  hdulist[0].header.set('BMAJ',BMAJ[initial_channel]/3600.0)
  hdulist[0].header.set('BMIN',BMIN[initial_channel]/3600.0)
  hdulist[0].header.set('BPA',BPA[initial_channel])
  hdu_aux = hdulist[0]
  hdu_aux.writeto('aux_cubes.fits',clobber=True,output_verify='fix')
  if ConfigFile['SaveCollapsedLines']:
    os.system('cp aux_cubes.fits '+OutputFolder+'/'+EL_name+'_collapsed.fits')
  f.show_contour('aux_cubes.fits', colors='aqua',levels=levels,linewidths=lws)
  c = SkyCoord(ra=ra2, dec=dec2, frame='icrs',unit=(u.deg, u.deg))
  f.add_scalebar(1.0/3600)
  f.scalebar.set_label('1"')
  f.scalebar.set_color('white')
  plt.text(0.5, 0.95,EL_name, horizontalalignment='center',verticalalignment='top',transform=plt.gca().transAxes,fontsize=40,color='black',bbox={'facecolor':'white', 'alpha':1.0, 'pad':2})

  f.recenter(c.ra.degree,c.dec.degree,radius=5.0/3600.0)
  f.set_theme('publication')
  f.axis_labels.set_font(size=20)
  f.tick_labels.set_font(size=15)
  f.axis_labels.set_ypad(-20)
  plt.savefig(output.replace('.pdf','_contours.pdf'),dpi=100)
  plt.close()
fitted_gaussian.write('\n')
isPS.write('\n')
fitted_gaussian.close()
isPS.close()

