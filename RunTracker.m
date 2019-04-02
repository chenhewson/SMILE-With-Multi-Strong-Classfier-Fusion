%%
% Demo for paper--Kaihua Zhang, Huihui song, 'Real-time Visual Tracking via
% Online Weighted Multiple Instance Learning', Pattern Recongtion.
% Author: Kaihua Zhang, Dept. of Computing, HK PolyU.
% Email: zhkhua@gmail.com
% Date: 8/8/1011
%%
clc;clear all;close all;

rand('state',0);% �ָ�����������������״̬
%----------------------------------
% The video sequences can be download from Boris's homepage
% http://vision.ucsd.edu/~bbabenko/project_miltrack.shtml
%----------------------------------
%----------------------------------
%addpath('./data');
path='F:\ѧϰ����\ѧϰС��\data\Benchmark\Football\img';
impath=[path,'\*.jpg'];
addpath(path);
%----------------------------------
load init.txt;
initstate = init;%initial tracker
%----------------------------Set path
%img_dir = dir('./data/*.jpg');
img_dir = dir(impath);
%-----------------------------The object position in the first frame
% x = initstate(1);% x axis at the Top left corner
% y = initstate(2);% y axis at the Top left corner
% w = initstate(3);% width of the rectangle
% h = initstate(4);% height of the rectangle
num = length(img_dir);% number of frames
%%5���뾶����,ǿ����������
rad=[5.0,4.0,4.5,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0];%ԭʼΪ4.0
srchwinsz=[26,28,27,25,25,25,25,25,25,25];%ԭʼΪ25
clfnum=2;
furtnum=150000;

for key=1:clfnum
    %% Parameter Settings
    trparams.init_negnumtrain = 50;%number of trained negative samples
    trparams.init_postrainrad = rad(key);%radical scope of positive samples; boy 8����������̽��뾶cxs��
    trparams.initstate = initstate;% object position [x y width height]
    trparams.srchwinsz = srchwinsz(key);% size of search window; boy 35���������뾶��cxs��
    %-------------------------
    % classifier parameters
    clfparams.width = trparams.initstate(3);
    clfparams.height= trparams.initstate(4);
    %-------------------------
    % feature parameters:number of rectangle
    ftrparams.minNumRect = 2;
    ftrparams.maxNumRect = 4;
    %-------------------------
    lRate = 0.85;% learning rate parameter ; 0.7 for biker1
    %-------------------------
    M = 150;% number of all weak classifiers in feature pool
    numSel = 15; % number of selected weak classifier ������ѡ�����ĵ���������������
    isin=0;
    rsconf=0.0;
    csconf=0.0;
    %-------------------------Initialize the feature mean and variance
    posx(key).mu = zeros(M,1);% mean of positive features
    negx(key).mu = zeros(M,1);
    posx(key).sig= ones(M,1);% variance of positive features
    negx(key).sig= ones(M,1);
    %-------------------------
    %compute feature template
    [ftr.px,ftr.py,ftr.pw,ftr.ph,ftr.pwt] = HaarFtr(clfparams,ftrparams,M);
    %% initilize the first frame
    %---------------------------
    img = imread(img_dir(1).name);
    img = double(img(:,:,1));  %ͼ��ĵ�һͨ��
    [rowz,colz] = size(img);
    %---------------------------
    %compute sample templates ��������ģ�壨������һ֡��Ԥ����׼����ÿһ֡���������Ҫ���㵱ǰ֡������ģ��cxs��
    posx(key).sampleImage = sampleImg(img,initstate,trparams.init_postrainrad,0,furtnum);
    negx(key).sampleImage = sampleImg(img,initstate,2*trparams.srchwinsz,1.5*trparams.init_postrainrad,trparams.init_negnumtrain);
    %--------extract haar features ����������ȡ
    iH = integral(img);%Compute integral image 
    selector = 1:M;  % select all weak classifier in pool ѡ�����������������
    posx(key).prob=nn(img);
    posx(key).feature = getFtrVal(iH,posx(key).sampleImage,ftr,selector,posx(key).prob);
    negx(key).feature = getFtrVal(iH,negx(key).sampleImage,ftr,selector,1.0);
    %--------Update the weak classifiers ���������ĸ���
    [posx(key).mu,posx(key).sig,negx(key).mu,negx(key).sig] = weakClfUpdate(posx(key),negx(key),posx(key).mu,posx(key).sig,negx(key).mu,negx(key).sig,lRate);% update distribution parameters ���·ֲ�����
    posx(key).pospred = weakClassifier(posx(key),negx(key),posx(key),selector);% Weak classifiers designed by positive samples
    negx(key).negpred = weakClassifier(posx(key),negx(key),negx(key),selector);% ... by negative samples
    %----------------------------------------------weight of the positive instance   
    %posx(key).w = exp(-((posx(key).sampleImage.sx-initstate(1)).^2+(posx(key).sampleImage.sy-initstate(2)).^2));
    posx(key).w = posx(key).prob;
    %-----------------------------------Feature selection
    selector = clfWMilBoostUpdate(posx(key),negx(key),numSel);%������cxs��
end
%--------------------------------------------------------
%--------------------------------(���Ĳ���cxs��


%% Start tracking
for i = 2:num
    img1 = imread(img_dir(i).name);
    img = double(img1(:,:,1));% Only utilize one channel of image��ͨ��
    detectx.sampleImage = sampleImg(img,initstate,trparams.srchwinsz,0,furtnum);
    iH = integral(img);%Compute integral image
    detectx.feature = getFtrVal(iH,detectx.sampleImage,ftr,selector,1.0);
    %----------------------------------��5����ͬ��posx(key),negx(key)��selector��ѵ�����õ�5��r�������cxs�Ľ�˼·)
    for key=1:clfnum
        r = 0;
        r = weakClassifier(posx(key),negx(key),detectx,selector);% compute the classifier for all samples��r��һ������,��15�У���ʾ15��ѡ�����������ķ�����cxs��
        prob(key).rsum = sum(r);% linearly combine the weak classifier in r to the strong classifier prob ��r�е���������������ϵ�ǿ�����������У�����r�������cxs��
        %-------------------------------------
        [c(key),index(key)] = max(prob(key).rsum);%c��ʾ���ֵ�Ƕ��٣�index��ʾ���ֵ��λ��
       
    end
    fprintf('c=%f\n',c);
    fprintf('index=%f\n',index);
    %��ѡ��c������ֵ��Ȼ��������ֵ���±�key�õ���index��key������λ�á�
    
    [value,position]=max(c);
    fprintf('maxvalue=%f\n',value);
    fprintf('maxvalue''s position=%d\n',position);
    fprintf('window index=%d\n',index(position));
    %fprintf('%d',);
    %�Ľ�˼·����5����ͬ�İ뾶ѵ����������ȡc���ģ�cxs��
    %-------------------------------------
    x = detectx.sampleImage.sx(index(position));
    y = detectx.sampleImage.sy(index(position));
    w = detectx.sampleImage.sw(index(position));
    h = detectx.sampleImage.sh(index(position));
    initstate = [x y w h];
    %-----------------------------------------Show the tracking result
    imshow(uint8(img1));
    rectangle('Position',initstate,'LineWidth',4,'EdgeColor','r');
    text(5, 18, strcat('#',num2str(i)), 'Color','y', 'FontWeight','bold', 'FontSize',20);
    set(gca,'position',[0 0 1 1]); 
    pause(0.00001); 
    for key = 1:clfnum;
        trparams.srchwinsz=srchwinsz(key);
        posx(key).prob=nn(img);
        %------------------------------------------���㵱ǰ֡������ģ�����ڶ���һ֡����Ԥ�⣨cxs��    
        posx(key).sampleImage = sampleImg(img,initstate,rad(key),0,furtnum);
        negx(key).sampleImage = sampleImg(img,initstate,1.5*trparams.srchwinsz,4+rad(key),trparams.init_negnumtrain);
        %------------------------------------------------weight of the positive instance    
        %posx(key).w = exp(-((posx(key).sampleImage.sx-initstate(1)).^2+(posx(key).sampleImage.sy-initstate(2)).^2));   
        posx(key).w = posx(key).prob;
        %-----------------------------------    
        %--------------------------------------------------Update all the features in pool
        selector = 1:M;
        posx(key).prob=nn(img);
        posx(key).feature = getFtrVal(iH,posx(key).sampleImage,ftr,selector,posx(key).prob);
        negx(key).feature = getFtrVal(iH,negx(key).sampleImage,ftr,selector,1.0);
        %--------------------------------------------------
        [posx(key).mu,posx(key).sig,negx(key).mu,negx(key).sig] = weakClfUpdate(posx(key),negx(key),posx(key).mu,posx(key).sig,negx(key).mu,negx(key).sig,lRate);% update distribution parameters
        posx(key).pospred = weakClassifier(posx(key),negx(key),posx(key),selector);
        negx(key).negpred = weakClassifier(posx(key),negx(key),negx(key),selector);
        %--------------------------------------------------
        selector = clfWMilBoostUpdate(posx(key),negx(key),numSel);% select the most discriminative weak classifiers ѡ����߱����������������cxs��

    end
end
%%