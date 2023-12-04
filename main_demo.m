close all; clear; clc;
addpath(genpath('./codes/'));

seed = 0;
rng('default');
rng(seed);
param.seed = seed;

db = {'MIRFLICKR','IAPRTC-12','NUSWIDE10','MIRFLICKR_deep','NUSWIDE21_deep'};
hashmethods = {'MIEH'};
loopnbits = [8 16 32 64 128];

param.top_R = 0;
param.top_K = 2000;
param.pr_ind = [1:50:1000,1001];
param.pn_pos = [1:100:2000,2000];

for dbi = 1%:5
    db_name = db{dbi}; param.db_name = db_name;
    
    % diary(['commandWindow_',db_name,'.txt']);
    % diary on;
    
    % load dataset
    load(['./datasets/',db_name,'.mat']);
    
    
    if strcmp(db_name, 'MIRFLICKR')
        X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
        R = randperm(size(L,1));
        queryInds = R(1:2000);
        sampleInds = R(2001:end);
        XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
        XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
        clear X Y L I_tr I_te T_tr T_te L_tr L_te
    
    elseif strcmp(db_name, 'IAPRTC-12')
        clear V_tr V_te
        X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
        R = randperm(size(L,1));
        queryInds = R(1:2000);
        sampleInds = R(2001:end);
        XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
        XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
        clear X Y L I_tr I_te T_tr T_te L_tr L_te V_tr V_te
    
    elseif strcmp(db_name, 'NUSWIDE10')
        X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
        R = randperm(size(L,1));
        queryInds = R(1:2000);
        sampleInds = R(2001:end);
        XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
        XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
        clear X Y L I_tr I_te T_tr T_te L_tr L_te
    
    elseif strcmp(db_name, 'MIRFLICKR_deep')
        R = randperm(size(X,1));
        queryInds = R(1:2000);
        sampleInds = R(2001:end);
        XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
        XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
        clear X Y L Y_pca
    
    elseif strcmp(db_name, 'NUSWIDE21_deep')
        R = randperm(size(X,1));
        queryInds = R(1:2000);
        sampleInds = R(2001:end);
        X = double(X);
        XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
        XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
        clear X Y L Latt500 XAll category_name
    end
    
    %% Label Format
    if isvector(LTrain)
        LTrain = sparse(1:length(LTrain), double(LTrain), 1); LTrain = full(LTrain);
        LTest = sparse(1:length(LTest), double(LTest), 1); LTest = full(LTest);
    end
    
    
    %% Methods
    eva_info = cell(length(hashmethods),length(loopnbits));
    
    for ii =1:length(loopnbits)
        fprintf('======%s: start %d bits encoding======\n\n',db_name,loopnbits(ii));
        param.nbits = loopnbits(ii);
        for jj = 1:length(hashmethods)
            
            switch(hashmethods{jj})
                case 'MIEH'
                    fprintf('......%s start...... \n\n', 'MIEH');
                    MIEHparam = param;
                    eva_info_ = evaluate_MIEH(XTrain,YTrain,LTrain,XTest,YTest,LTest,MIEHparam);
                % case 'XXXX'
                %     fprintf('......%s start...... \n\n', 'XXXX');
                %     XXXXparam = param;
                %     eva_info_ = evaluate_XXXX(XTrain,YTrain,LTrain,XTest,YTest,LTest,XXXXparam);
            end
            eva_info{jj,ii} = eva_info_;
            clear eva_info_
        end
    end
    
    
    %% MAP
    % for ii = 1:length(loopnbits)
    %     for jj = 1:length(hashmethods)
    %         % MAP
    %         Image_VS_Text_MAP{jj,ii} = eva_info{jj,ii}.Image_VS_Text_MAP;
    %         Text_VS_Image_MAP{jj,ii} = eva_info{jj,ii}.Text_VS_Image_MAP;
    % 
    %         % NDCG
    %         Image_VS_Text_NDCG{jj,ii} = eva_info{jj,ii}.Image_VS_Text_NDCG;
    %         Text_VS_Image_NDCG{jj,ii} = eva_info{jj,ii}.Text_VS_Image_NDCG;
    % 
    %         % Precision VS Recall
    %         Image_VS_Text_recall{jj,ii} = eva_info{jj,ii}.Image_VS_Text_recall(param.pr_ind)';
    %         Image_VS_Text_precision{jj,ii} = eva_info{jj,ii}.Image_VS_Text_precision(param.pr_ind)';
    %         Text_VS_Image_recall{jj,ii} = eva_info{jj,ii}.Text_VS_Image_recall(param.pr_ind)';
    %         Text_VS_Image_precision{jj,ii} = eva_info{jj,ii}.Text_VS_Image_precision(param.pr_ind)';
    % 
    %         % Top number Precision
    %         Image_To_Text_Precision{jj,ii} = eva_info{jj,ii}.Image_To_Text_Precision(param.pn_pos)';
    %         Text_To_Image_Precision{jj,ii,:} = eva_info{jj,ii}.Text_To_Image_Precision(param.pn_pos)';
    % 
    %         % time
    %         trainT{jj,ii} = eva_info{jj,ii}.trainT;
    %         %compressT{jj,ii} = eva_info{jj,ii}.compressT;
    %         %testT{jj,ii} = eva_info{jj,ii}.testT;
    %     end
    % end

    % diary off;
    
    
    %% Save
    % result_URL = './results/';
    % if ~isdir(result_URL)
    %     mkdir(result_URL);
    % end
    % result_name = [result_URL 'final_' db_name '_result' '.mat'];
    % 
    % save(result_name,'eva_info','loopnbits','hashmethods','param',...
    %     'XTrain','XTest','YTrain','YTest','LTrain','LTest',...
    %     'Image_VS_Text_MAP','Text_VS_Image_MAP',...%
    %     'Image_VS_Text_NDCG','Text_VS_Image_NDCG',...
    %     'trainT','Image_To_Text_Precision','Text_To_Image_Precision',...
    %     'Image_VS_Text_recall','Image_VS_Text_precision','Text_VS_Image_recall','Text_VS_Image_precision','-v7.3');

end