
clc
clear all
close all


Data = '' % Directory which data is located
hofiles= {'ms' 25;  
          'n' 25;
                    };


lag =1;
dim =2;
e=0;

number_of_epochs = 7;
epoch = 30;
duration = epoch *200 ; 






F_Alpha = [];           
F_Beta = [];           
F_Theta = [];           
F_Delta = [];           
F_Gamma = [];
F_Data = [];

F= [];
features= [];
features_Gama=[];
features_Delta=[];
features_Theta=[];
features_Alpha =[];
features_Beta =[];
features_Data=[];
for i22=1:size(hofiles,1)
    d=0;

    for j=1:cell2mat(hofiles(i22,2))
    d = load([Data char(hofiles(i22,1)) '' num2str(j) '.mat']);
    
    f =struct2cell(d);
indexToRemove = [];
for i = 1:length(f)
    if ~isa(f{i}, 'double')
        indexToRemove = i;
        break;
    end
end
if ~isempty(indexToRemove)
    f(indexToRemove) = [];
end

    xv=0;
    for i2 = 1:number_of_epochs  %% Epoch size
    xv =xv +2000;
    features_Alpha=[];
    features_Beta=[];
    features_Delta=[];
    features_Theta=[];
    features_Gama=[];
    features_Data=[];

    for jj = 1:size(f,1)-1  %% Channels
            dd =cell2mat(f(jj));
            ignal = dd(xv:xv+duration);
            
            
            
            
            SIGNALS = Signal;
            fs = 200; % Sampling frequency
            Alpha1 = 8;
            Alpha2 = 12;
            [b, a] = butter(4, [Alpha1, Alpha2] / (fs / 2), 'bandpass');
            Sig_alpha = filtfilt(b, a, Signal); % Apply filterSign
            % 
            
            
            Delta1 = 0.5;
            Delta2 = 4;
            [b, a] = butter(4, [Delta1, Delta2] / (fs / 2), 'bandpass');
            Sig_delta = filtfilt(b, a, Signal); % Apply filterSign
            
            
            
            Theta1 = 4;
            Theta2 = 8;
            [b, a] = butter(4, [Theta1, Theta2] / (fs / 2), 'bandpass');
            Sig_theta = filtfilt(b, a, Signal); % Apply filterSign
            
            
            Beta1 = 12;
            Beta2 = 30;
            [b, a] = butter(4, [Beta1, Beta2] / (fs / 2), 'bandpass');
            Sig_beta = filtfilt(b, a, Signal); % Apply filterSign
            
            
            
            Gama1 = 30;
            Gama2 = 99;
            [b, a] = butter(4, [Gama1, Gama2] / (fs / 2), 'bandpass');
            Sig_gama = filtfilt(b, a, Signal); % Apply filterSign



%%%%%%% ALPHA Band %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_n_alpha = Sig_alpha(1:end-T);  % Values at time n
x_n1_alpha = Sig_alpha(T+1:end);   % Values at time n+1
mean_x = mean(Sig_alpha);
x_diff = x_n1_alpha - x_n_alpha;
SD1_Alpha = sqrt(mean(x_diff.^2) / 2);
SD2_Alpha = sqrt(2 * var(Sig_alpha) - SD1_Alpha^2);
ellipse_area_Alpha = pi * SD1_Alpha * SD2_Alpha;
SD12_Alpha = SD1_Alpha/SD2_Alpha ;

%%%%%%% BETA Band  %%%%%%%%%%%%%%%%%%%%%
x_n_beta = Sig_beta(1:end-T);  % Values at time n
x_n1_beta = Sig_beta(T+1:end);   % Values at time n+1
mean_x = mean(Sig_beta);
x_diff = x_n1_beta - x_n_beta;
SD1_Beta = sqrt(mean(x_diff.^2) / 2);
SD2_Beta = sqrt(2 * var(Sig_beta) - SD1_Beta^2);
ellipse_area_Beta = pi * SD1_Beta * SD2_Beta;
SD12_Beta = SD1_Beta/SD2_Beta ;

%%%%%%  THETA Band %%%%%%%%%%%%%%%%%%%%
x_n_theta = Sig_theta(1:end-T);  % Values at time n
x_n1_theta = Sig_theta(T+1:end);   % Values at time n+1
mean_x = mean(Sig_theta);
x_diff = x_n1_theta - x_n_theta;
SD1_Theta = sqrt(mean(x_diff.^2) / 2);
SD2_Theta  = sqrt(2 * var(Sig_theta) - SD1_Theta^2);
ellipse_area_Theta  = pi * SD1_Theta  * SD2_Theta ;
SD12_Theta = SD1_Theta /SD2_Theta ;
%%%%%   DELTA Band  %%%%%%%%%%%%%%%%%%%%%%
x_n_delta = Sig_delta(1:end-T);  % Values at time n
x_n1_delta = Sig_delta(T+1:end);   % Values at time n+1
mean_x = mean(Sig_delta);
x_diff = x_n1_delta - x_n_delta;
SD1_Delta = sqrt(mean(x_diff.^2) / 2);
SD2_Delta   = sqrt(2 * var(Sig_delta) - SD1_Delta^2);
ellipse_area_Delta  = pi * SD1_Delta   * SD2_Delta  ;
SD12_Delta  = SD1_Delta  /SD2_Delta  ;

%%%%%%  GAMMA Band   %%%%%%%%%%%%%%%%%%%%%%%%
x_n_gama = Sig_gama(1:end-T);  % Values at time n
x_n1_gama = Sig_gama(T+1:end);   % Values at time n+1
mean_x = mean(Sig_gama);
x_diff = x_n1_gama - x_n_gama;
SD1_Gama  = sqrt(mean(x_diff.^2) / 2);
SD2_Gama  = sqrt(2 * var(Sig_gama) - SD1_Gama^2);
ellipse_area_Gama  = pi * SD1_Gama    * SD2_Gama   ;
SD12_Gama   = SD1_Gama   /SD2_Gama   ;
%%%%%%%%%   Whole Signal    %%%%%%%%%%%%%%%%%%%%%
x_n_data = Signal(1:end-T);  % Values at time n
x_n1_data = Signal(T+1:end);   % Values at time n+1
mean_x = mean(Signal);
x_diff_data = x_n1_data - x_n_data;
SD1_data  = sqrt(mean(x_diff_data.^2) / 2);
SD2_data  = sqrt(2 * var(Signal) - SD1_data^2);
ellipse_area_data  = pi * SD1_data    * SD2_data   ;
SD12_data   = SD1_data   /SD2_data   ;

%%%%%%%%%%% Recording features %%%%%%%%%%%%%%%%%%%
feat_Alpha = [SD1_Alpha,SD2_Alpha,SD12_Alpha,ellipse_area_Alpha];
features_Alpha = [features_Alpha,feat_Alpha];

feat_Beta = [SD1_Beta,SD2_Beta,SD12_Beta,ellipse_area_Beta];
features_Beta = [features_Beta,feat_Beta];

feat_Theta = [SD1_Theta,SD2_Theta,SD12_Theta,ellipse_area_Theta];
features_Theta = [features_Theta,feat_Theta];

feat_Delta= [SD1_Delta,SD2_Delta,SD12_Delta,ellipse_area_Delta];
features_Delta = [features_Delta,feat_Delta];


feat_Gama= [SD1_Gama,SD2_Gama,SD12_Gama,ellipse_area_Gama];
features_Gama = [features_Gama,feat_Gama];


feat_Data= [SD1_data,SD2_data,SD12_data,ellipse_area_data];
features_Data = [features_Data,feat_Data];






end
% exportgraphics(gcf, 'EEG_Poincare_Epochs_HighQuality.png', 'Resolution', 300);  % Save as PNG at 300 DPI


F_Alpha = [F_Alpha; features_Alpha];           
F_Beta  = [F_Beta; features_Beta];           
F_Theta = [F_Theta; features_Theta];           
F_Delta = [F_Delta; features_Delta];           
F_Gamma = [F_Gamma; features_Gama];    
F_Data = [F_Data; features_Data];    

    end
    end
    end
  


MS = ones(25*number_of_epochs,1);
NO = zeros(25*number_of_epochs,1);
label = [MS;NO];

Dataset_Whole = [F_Data,label];
Dataset_Alpha =[F_Alpha,label];
Dataset_Beta =[F_Beta,label];
Dataset_Theta =[F_Theta,label];
Dataset_Delta =[F_Delta,label];
Dataset_Gamma =[F_Gamma,label];


save('Dataset_Whole_30s.mat', 'Dataset_Whole');
save('Dataset_Alpha_30s.mat', 'Dataset_Alpha');
save('Dataset_Beta_30s.mat', 'Dataset_Beta');
save('Dataset_Theta_30s.mat', 'Dataset_Theta');
save('Dataset_Delta_30s.mat', 'Dataset_Delta');
save('Dataset_Gamm_30s.mat', 'Dataset_Gamma');


