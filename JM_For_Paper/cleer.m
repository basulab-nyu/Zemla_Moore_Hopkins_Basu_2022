clear;
% fs = 32556;
% dt = 1000/fs;
sizeTitle = 30;
sizeFigFont = 18;
sizeLabels = 24;
lw = 'Linewidth';

set(0,'defaultaxesfontsize',12);
set(0,'defaulttextfontsize',24);
% set(0,'defaultaxesfontsize',26);
% set(0,'defaulttextfontsize',28);
set(0,'defaultlinelinewidth',1);
set(0,'defaultlinemarkersize',20)

f = figure;
whitebg([1 1 1]);
close(f);
clear f;


green =     [0.0 0.75 0.0];
lblue =     [0.2 0.8 1.0];
orange =    [1.0 0.4 0.0];
yellow =    [1.0 0.65 0.0];
blue2 =     [0.2 0.5 1.0];
green2 =    [0.0 0.9 0];
black2 =    [0.5 0.5 0.5];
red2 =      [1.0 0.45 0.45];
orange2 =   [1.0 0.6 0.0];
gray =      [0.7 0.7 0.7];
purple =    [0.7 0.0 0.7];

dgray =     [0.2 0.2 0.2];
brick =     [0.9 0.0 0.0];
dbrick =    [0.75 0.0 0.0];
dblue =     [0.0 0.0 0.9];

co = [0 0 1;
      0 0.5 0;
      1 0 0;
      0 0.75 0.75;
      0.75 0 0.75;
      0.75 0.75 0;
      0.25 0.25 0.25];
set(groot,'defaultAxesColorOrder',co);
set(groot,'DefaultFigureColormap',jet)
% close all;
