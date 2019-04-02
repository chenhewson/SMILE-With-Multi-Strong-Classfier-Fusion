%%
% Demo for paper--Kaihua Zhang, Huihui song, 'Real-time Visual Tracking via
% Online Weighted Multiple Instance Learning', Pattern Recongtion.
% Author: Kaihua Zhang, Dept. of Computing, HK PolyU.
% Email: zhkhua@gmail.com
% Date: 8/8/1011
%%
clc;clear all;close all;

rand('state',0);% 恢复到最初产生随机数的状态
%----------------------------------
% The video sequences can be download from Boris's homepage
% http://vision.ucsd.edu/~bbabenko/project_miltrack.shtml
%----------------------------------
%----------------------------------
%addpath('./data');
path='F:\学习资料\学习小组\data\Benchmark\Football\img';
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
%%5个半径数组,强分类器个数
rad=[5.0,4.0,4.5,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0,4.0];%原始为4.0
srchwinsz=[26,28,27,25,25,25,25,25,25,25];%原始为25
clfnum=2;
furtnum=150000;

for key=1:clfnum
    %% Parameter Settings
    trparams.init_negnumtrain = 50;%number of trained negative samples
    trparams.init_postrainrad = rad(key);%radical scope of positive samples; boy 8（正样本的探测半径cxs）
    trparams.initstate = initstate;% object position [x y width height]
    trparams.srchwinsz = srchwinsz(key);% size of search window; boy 35正包搜索半径（cxs）
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
    numSel = 15; % number of selected weak classifier （已挑选出来的的弱分类器数量）
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
    img = double(img(:,:,1));  %图像的第一通道
    [rowz,colz] = size(img);
    %---------------------------
    %compute sample templates 计算样本模板（用于下一帧的预测做准备，每一帧分类结束都要计算当前帧的样本模板cxs）
    posx(key).sampleImage = sampleImg(img,initstate,trparams.init_postrainrad,0,furtnum);
    negx(key).sampleImage = sampleImg(img,initstate,2*trparams.srchwinsz,1.5*trparams.init_postrainrad,trparams.init_negnumtrain);
    %--------extract haar features 哈尔特征提取
    iH = integral(img);%Compute integral image 
    selector = 1:M;  % select all weak classifier in pool 选择池中所有弱分类器
    posx(key).prob=nn(img);
    posx(key).feature = getFtrVal(iH,posx(key).sampleImage,ftr,selector,posx(key).prob);
    negx(key).feature = getFtrVal(iH,negx(key).sampleImage,ftr,selector,1.0);
    %--------Update the weak classifiers 弱分类器的更新
    [posx(key).mu,posx(key).sig,negx(key).mu,negx(key).sig] = weakClfUpdate(posx(key),negx(key),posx(key).mu,posx(key).sig,negx(key).mu,negx(key).sig,lRate);% update distribution parameters 更新分布参数
    posx(key).pospred = weakClassifier(posx(key),negx(key),posx(key),selector);% Weak classifiers designed by positive samples
    negx(key).negpred = weakClassifier(posx(key),negx(key),negx(key),selector);% ... by negative samples
    %----------------------------------------------weight of the positive instance   
    %posx(key).w = exp(-((posx(key).sampleImage.sx-initstate(1)).^2+(posx(key).sampleImage.sy-initstate(2)).^2));
    posx(key).w = posx(key).prob;
    %-----------------------------------Feature selection
    selector = clfWMilBoostUpdate(posx(key),negx(key),numSel);%（存疑cxs）
end
%--------------------------------------------------------
%--------------------------------(更改部分cxs）


%% Start tracking
for i = 2:num
    img1 = imread(img_dir(i).name);
    img = double(img1(:,:,1));% Only utilize one channel of image单通道
    detectx.sampleImage = sampleImg(img,initstate,trparams.srchwinsz,0,furtnum);
    iH = integral(img);%Compute integral image
    detectx.feature = getFtrVal(iH,detectx.sampleImage,ftr,selector,1.0);
    %----------------------------------用5个不同的posx(key),negx(key)，selector来训练，得到5个r，求最大（cxs改进思路)
    for key=1:clfnum
        r = 0;
        r = weakClassifier(posx(key),negx(key),detectx,selector);% compute the classifier for all samples（r是一个矩阵,有15行，表示15个选择器挑出来的分类器cxs）
        prob(key).rsum = sum(r);% linearly combine the weak classifier in r to the strong classifier prob 将r中的弱分类器线性组合到强分类器概率中（矩阵r的列求和cxs）
        %-------------------------------------
        [c(key),index(key)] = max(prob(key).rsum);%c表示最大值是多少，index表示最大值的位置
       
    end
    fprintf('c=%f\n',c);
    fprintf('index=%f\n',index);
    %挑选出c中最大的值，然后根据最大值的下标key得到的index（key）就是位置。
    
    [value,position]=max(c);
    fprintf('maxvalue=%f\n',value);
    fprintf('maxvalue''s position=%d\n',position);
    fprintf('window index=%d\n',index(position));
    %fprintf('%d',);
    %改进思路，用5个不同的半径训练五个结果，取c最大的（cxs）
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
        %------------------------------------------计算当前帧的样本模板用于对下一帧进行预测（cxs）    
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
        selector = clfWMilBoostUpdate(posx(key),negx(key),numSel);% select the most discriminative weak classifiers 选择最具辨别力的弱分类器（cxs）

    end
end
%%