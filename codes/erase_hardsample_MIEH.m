function id_nh = erase_hardsample(XTrain,YTrain,LTrain,param)

    [n,c] = size(LTrain);
    
    XP = (XTrain'*XTrain+param.xi*eye(size(XTrain,2))) \ (XTrain'*LTrain);
    YP = (YTrain'*YTrain+param.xi*eye(size(YTrain,2))) \ (YTrain'*LTrain);
    [~,LTrain_px] = max(XTrain*XP,[],2);
    [~,LTrain_py] = max(YTrain*YP,[],2);
    LTrain_px = sparse(1:n, double(LTrain_px), 1); LTrain_px = full(LTrain_px);
    LTrain_py = sparse(1:n, double(LTrain_py), 1); LTrain_py = full(LTrain_py);
    if size(LTrain_px,2)<c
        LTrain_px(:,size(LTrain_px,2)+1:c) = 0;
    end
    if size(LTrain_py,2)<c
        LTrain_py(:,size(LTrain_py,2)+1:c) = 0;
    end
    idx = sum(LTrain_px.*LTrain,2)==0; %error
    idy = sum(LTrain_py.*LTrain,2)==0;
    id_nh = find(idx.*idy~=1); % not hard
end