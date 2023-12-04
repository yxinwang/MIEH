function evaluation_info=evaluate_MIEH(XTrain,YTrain,LTrain,XTest,YTest,LTest,param)

    %% ----------------- Settings -------------------
    %normalization
    % param.ifnormalize1 = 0;
    % param.ifnormalize2 = 0;
    param.ifkernel = 1;
    param.nAnchors = 1500;
    
    %common latent space
    param.max_iter_V = 10;
    param.etax = 0.1; param.etay = 0.2; param.etal = 0.7;
    param.theta = 0.01; %regularization
    param.dV = param.nbits; %dim of common space V
    
    %class proxy
    param.max_iter_proxy = 10;
    param.mu = 10; %proxy seperation
    
    %clustering
    param.ngroup = 10; %cluster

    %hash code learning
    param.max_iter_hash = 10;
    param.alpha1 = 1; param.alpha2 = 1; %instance-instance
    param.beta1 = 1; %regularization
    param.beta2 = 1; %class proxy
    param.beta3 = 1; %latent space
    
    %hash function learning
    param.xi = 0.01;
    
    if strcmp(param.db_name, 'MIRFLICKR')
        param.LmdSet = [4 4 4 4 8];
        param.n_sel = 128;
    elseif strcmp(param.db_name, 'IAPRTC-12')
        param.LmdSet = [4 4 8 8 8];
        param.n_sel = 256;
    elseif strcmp(param.db_name, 'NUSWIDE10')
        param.ifnormalize2 = 1;
        param.LmdSet = [4 4 4 8 8];
        param.n_sel = 128;
    elseif strcmp(param.db_name, 'MIRFLICKR_deep')
        param.ifnormalize2 = 1;
        param.LmdSet = [4 4 4 4 8];
        param.n_sel = 128;
    elseif strcmp(param.db_name, 'IAPRTC-12_deep')
        param.ifnormalize2 = 1;
        param.LmdSet = [4 4 8 8 8];
        param.n_sel = 256;
    elseif strcmp(param.db_name, 'NUSWIDE21_deep')
        param.ifnormalize2 = 1;
        param.LmdSet = [4 4 4 8 8];
        param.n_sel = 8*param.nbits;
    end

    n = size(LTrain,1);
    
    % %normalize
    % if param.ifnormalize1
    %     XTrain = (XTrain-repmat(mean(XTrain,1),size(XTrain,1),1));
    %     YTrain = (YTrain-repmat(mean(YTrain,1),size(YTrain,1),1));
    %     XTest = (XTest-repmat(mean(XTrain,1),size(XTest,1),1));
    %     YTest = (YTest-repmat(mean(YTrain,1),size(YTest,1),1));
    % end
    % if param.ifnormalize2
    %     XTrain = NormalizeFea(XTrain,1); % row L2-normalized
    %     YTrain = NormalizeFea(YTrain,1);
    %     XTest = NormalizeFea(XTest,1);
    %     YTest = NormalizeFea(YTest,1);
    % end
    
    %% ----------------- Multiple Information Mining -------------------
    tic;
    %kernelization
    if param.ifkernel
        anchor_idx = randsample(n, param.nAnchors);
        XTest = RBF_fast(XTest,XTrain(anchor_idx,:));
        XTrain = RBF_fast(XTrain,XTrain(anchor_idx,:));
        anchor_idx = randsample(n, param.nAnchors);
        YTest = RBF_fast(YTest,YTrain(anchor_idx,:));
        YTrain = RBF_fast(YTrain,YTrain(anchor_idx,:));
    end
    
    %hard example erasing
    id_nh = erase_hardsample_MIEH(XTrain,YTrain,LTrain,param);
    
    % latent common space
    disp('latent space learning');
    V = generate_comspace_MIEH(XTrain,YTrain,LTrain,param);
    
    % semantic class proxy
    disp('class proxy');
    Yg = generate_proxy_MIEH(LTrain,param);
    
    %clustering
    disp('clustering');
    if param.ngroup == 1
        id_cl{1,1} = 1:n;
    else
        class = litekmeans([XTrain YTrain], param.ngroup);
        % rearrangement samples
        idx = cell(1,param.ngroup); id_cl = cell(1,param.ngroup);
        begin = 0;
        for gi = 1:param.ngroup
            idx{1,gi} = find(class == gi)';
            id_cl{1,gi} = begin+1:begin+length(idx{1,gi});
            begin = begin+length(idx{1,gi});
        end
        idx_all = cell2mat(idx(1,1:param.ngroup));
        clear class idx begin

        LTrain = LTrain(idx_all,:);
        XTrain = XTrain(idx_all,:);
        YTrain = YTrain(idx_all,:);
        V = V(idx_all,:);
    end
    
    evaluation_info.trainT0=toc;
    
    
    %% ----------------- Hash Codes Learning -------------------
    tic;
    B = train_MIEH(LTrain,V,Yg,id_cl,param);
    evaluation_info.trainT1=toc;
    
    
    %% ----------------- Hash Functions Learning -------------------
    tic;
    ids = id_nh(randperm(length(id_nh),param.n_sel));
    Ss_ = (LTrain(id_nh,:)*LTrain(ids,:)'>=1); Ss_ = 2*Ss_-1;
    XW = (XTrain(id_nh,:)'*XTrain(id_nh,:)+param.xi*eye(size(XTrain(id_nh,:),2)))\(XTrain(id_nh,:)'*B(id_nh,:)+XTrain(id_nh,:)'*Ss_*B(ids,:)*param.xi*param.nbits)/(eye(param.nbits)+B(ids,:)'*B(ids,:)*param.xi);
    YW = (YTrain(id_nh,:)'*YTrain(id_nh,:)+param.xi*eye(size(YTrain(id_nh,:),2)))\(YTrain(id_nh,:)'*B(id_nh,:)+YTrain(id_nh,:)'*Ss_*B(ids,:)*param.xi*param.nbits)/(eye(param.nbits)+B(ids,:)'*B(ids,:)*param.xi);
    evaluation_info.trainT2=toc;
    
    
    %% ----------------- Out of samples -------------------
    tic;
    BxTrain = compactbit(B>=0);
    ByTrain = BxTrain;
    BxTest = compactbit(XTest*XW>0);
    ByTest = compactbit(YTest*YW>0);
    evaluation_info.compressT = toc;
    
    
    %% --------------------- Evaluation ---------------------
    tic;
    DHamm = hammingDist(BxTest, ByTrain);
    [~, orderH] = sort(DHamm, 2);
	evaluation_info.Image_VS_Text_MAP = mAP(orderH', LTrain, LTest);
    [evaluation_info.Image_VS_Text_precision, evaluation_info.Image_VS_Text_recall] = precision_recall(orderH', LTrain, LTest);
    evaluation_info.Image_To_Text_Precision = precision_at_k(orderH', LTrain, LTest,param.top_K);
    evaluation_info.Image_VS_Text_NDCG = ndcg_our(orderH,LTrain, LTest);
    
    DHamm = hammingDist(ByTest, BxTrain);
    [~, orderH] = sort(DHamm, 2);
    evaluation_info.Text_VS_Image_MAP = mAP(orderH', LTrain, LTest);
    [evaluation_info.Text_VS_Image_precision,evaluation_info.Text_VS_Image_recall] = precision_recall(orderH', LTrain, LTest);
    evaluation_info.Text_To_Image_Precision = precision_at_k(orderH', LTrain, LTest,param.top_K);
    evaluation_info.Text_VS_Image_NDCG = ndcg_our(orderH,LTrain, LTest);
    
    evaluation_info.testT = toc;
    
    evaluation_info.trainT = evaluation_info.trainT0+evaluation_info.trainT1+evaluation_info.trainT2;
    evaluation_info.param = param;
    
    fprintf('...i2t:%.4f,   t2i:%.4f\n', evaluation_info.Image_VS_Text_MAP, evaluation_info.Text_VS_Image_MAP);
end