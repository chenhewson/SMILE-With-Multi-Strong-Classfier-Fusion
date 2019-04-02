#include <FerNNClassifier.h>
#include "mex.h"
using namespace cv;
using namespace std;
 
void FerNNClassifier::read(const FileNode& file){
  ///Classifier Parameters
  valid = (float)file["valid"];
  ncc_thesame = (float)file["ncc_thesame"];
  thr_fern = (float)file["thr_fern"];
  thr_nn = (float)file["thr_nn"];
  thr_nn_valid = (float)file["thr_nn_valid"];
}
void FerNNClassifier::prepare(const vector<Size>& scales){
  acum = 0;
  //Initialize test locations for features
  int totalFeatures = nstructs*structSize;
  features = vector<vector<Feature> >(scales.size(),vector<Feature> (totalFeatures));
  RNG& rng = theRNG();
  float x1f,x2f,y1f,y2f;
  int x1, x2, y1, y2;
  for (int i=0;i<totalFeatures;i++){
      x1f = (float)rng;
      y1f = (float)rng;
      x2f = (float)rng;
      y2f = (float)rng;
      for (int s=0;s<scales.size();s++){
          x1 = x1f * scales[s].width;
          y1 = y1f * scales[s].height;
          x2 = x2f * scales[s].width;
          y2 = y2f * scales[s].height;
          features[s][i] = Feature(x1, y1, x2, y2);//��ν���رȽϣ�����ָ�������λ�á���������ģ������󲻸ı�
      }
 
  }
  //Thresholds
  thrN = 0.5*nstructs;
 
  //Initialize Posteriors
  for (int i = 0; i<nstructs; i++) {
      posteriors.push_back(vector<float>(pow(2.0,structSize), 0));
      pCounter.push_back(vector<int>(pow(2.0,structSize), 0));
      nCounter.push_back(vector<int>(pow(2.0,structSize), 0));
  }
}
void FerNNClassifier::trainNN(const vector<cv::Mat>& nn_examples){
  float conf,dummy;
  vector<int> y(nn_examples.size(),0);//v3����n��ֵΪi��Ԫ�ء�y����Ԫ�س�ʼ��Ϊ0
  y[0]=1;
  vector<int> isin;
  for (int i=0;i<nn_examples.size();i++){                          //  For each example
      NNConf(nn_examples[i],isin,conf,dummy);                      //  Measure Relative similarity��������ͼ��Ƭ������ģ��֮���������ƶ�conf
      if (y[i]==1 && conf<=thr_nn){                                //    if y(i) == 1 && conf1 <= tld.model.thr_nn % 0.65��ǩ�������������������ƶ�С��0.65 ������Ϊ�䲻����ǰ��Ŀ�꣬Ҳ���Ƿ�������ˣ���ʱ��Ͱ����ӵ���������
          if (isin[1]<0){                                          //      if isnan(isin(2))
              pEx = vector<Mat>(1,nn_examples[i]);                 //        tld.pex = x(:,i);
              continue;                                            //        continue;
          }                                                        //      end
          //pEx.insert(pEx.begin()+isin[1],nn_examples[i]);        //      tld.pex = [tld.pex(:,1:isin(2)) x(:,i) tld.pex(:,isin(2)+1:end)]; % add to model
          pEx.push_back(nn_examples[i]);
      }                                                            //    end
      if(y[i]==0 && conf>0.5)                                      //  if y(i) == 0 && conf1 > 0.5
        nEx.push_back(nn_examples[i]);                             //    tld.nex = [tld.nex x(:,i)];
 
  }                                                                 //  end
  acum++;
  printf("%d. Trained NN examples: %d positive %d negative\n",acum,(int)pEx.size(),(int)nEx.size());
}                                                                  //  end
 
/**
 * ����nn������ĺ��ģ�����ĸ��ݾ��Ǹ������ƶȵ�ֵ���ֵ�
 * rsconf ���ƶ�
 * csconf �������ƶ�
 */
void FerNNClassifier::NNConf(const Mat& example, vector<int>& isin,float& rsconf,float& csconf){
  /*Inputs:
   * -NN Patch
   * Outputs:
   * -Relative Similarity (rsconf), Conservative Similarity (csconf), In pos. set|Id pos set|In neg. set (isin)
   */
  isin=vector<int>(3,-1);
  if (pEx.empty()){ //if isempty(tld.pex) % IF positive examples in the model are not defined THEN everything is negative
      rsconf = 0; //    conf1 = zeros(1,size(x,2));
      csconf=0;
      return;
  }
  if (nEx.empty()){ //if isempty(tld.nex) % IF negative examples in the model are not defined THEN everything is positive
      rsconf = 1;   //    conf1 = ones(1,size(x,2));
      csconf=1;
      return;
  }
  Mat ncc(1,1,CV_32F);
  float nccP,csmaxP,maxP=0;
  bool anyP=false;
  int maxPidx,validatedPart = ceil(pEx.size()*valid);
  float nccN, maxN=0;
  bool anyN=false;
  for (int i=0;i<pEx.size();i++){
      matchTemplate(pEx[i],example,ncc,CV_TM_CCORR_NORMED);      // measure NCC to positive examples ��һ�����ƥ�䷨
      nccP=(((float*)ncc.data)[0]+1)*0.5;//������������������ƶ�
      if (nccP>ncc_thesame)
        anyP=true;
      if(nccP > maxP){
          maxP=nccP;
          maxPidx = i;
          if(i<validatedPart)
            csmaxP=maxP;
      }
  }
  for (int i=0;i<nEx.size();i++){
      matchTemplate(nEx[i],example,ncc,CV_TM_CCORR_NORMED);     //���㸺������������ƶ�
      nccN=(((float*)ncc.data)[0]+1)*0.5;
      if (nccN>ncc_thesame)
        anyN=true;
      if(nccN > maxN)
        maxN=nccN;
  }
  //set isin
  if (anyP) isin[0]=1;  //if he query patch is highly correlated with any positive patch in the model then it is considered to be one of them
  isin[1]=maxPidx;      //get the index of the maximall correlated positive patch
  if (anyN) isin[2]=1;  //if  the query patch is highly correlated with any negative patch in the model then it is considered to be one of them
  //Measure Relative Similarity������ƶ� = ��������������ƶ� / ����������������ƶ� + ��������������ƶȣ�
  float dN=1-maxN;
  float dP=1-maxP;
  rsconf = (float)dN/(dN+dP);
  //Measure Conservative Similarity
  dP = 1 - csmaxP;
  csconf =(float)dN / (dN + dP);
}
/**
 * �����µĸ����������¼���fern��������ֵ �� nn����������ֵ
 * ��ֵҪ�����еĸ����������Ŷ�Ҫ��
 */
void FerNNClassifier::evaluateTh(const vector<pair<vector<int>,int> >& nXT,const vector<cv::Mat>& nExT){
float fconf;
  for (int i=0;i<nXT.size();i++){
    fconf = (float) measure_forest(nXT[i].first)/nstructs;
    if (fconf>thr_fern)
      thr_fern=fconf;
}
  vector <int> isin;
  float conf,dummy;
  for (int i=0;i<nExT.size();i++){
      NNConf(nExT[i],isin,conf,dummy);
      if (conf>thr_nn)
        thr_nn=conf;//ȡ������������ƶ���Ϊ ������ڷ��������µ���ֵ
  }
  if (thr_nn>thr_nn_valid)
    thr_nn_valid = thr_nn;
}
 
void FerNNClassifier::show(){
  Mat examples((int)pEx.size()*pEx[0].rows,pEx[0].cols,CV_8U);
  double minval;
  Mat ex(pEx[0].rows,pEx[0].cols,pEx[0].type());
  for (int i=0;i<pEx.size();i++){
    minMaxLoc(pEx[i],&minval);
    pEx[i].copyTo(ex);
    ex = ex-minval;
    Mat tmp = examples.rowRange(Range(i*pEx[i].rows,(i+1)*pEx[i].rows));
    ex.convertTo(tmp,CV_8U);
  }
  imshow("Examples",examples);
  
}