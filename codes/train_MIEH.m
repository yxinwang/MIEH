function B = train_MIEH(LTrain,V,Yg,id_cl,param)
% code for "Multiple Information Embedded Hashing for Large-Scale Cross-Modal Retrieval", IEEE TCSVT

    % parameters
    max_iter = param.max_iter_hash;
    ngroup = param.ngroup;
    alpha1 = param.alpha1; alpha2 = param.alpha2;
    beta1 = param.beta1; beta2 = param.beta2; beta3 = param.beta3;
    lambda = param.LmdSet(log2(param.nbits)-2);
    n_sel = param.n_sel;
    nbits = param.nbits;
    
    n = size(LTrain,1);

    %% ----------- Efficient Discrete Optimization ---------------
    %initization
    G_opt = ones(n, nbits);
    G_opt(randn(n, nbits) < 0) = -1;
    B = ones(n, nbits);
    B(randn(n, nbits) < 0) = -1;
    
    %alternative updating
    fprintf('iteration ..\n');
    for epoch = 1:max_iter
        % fprintf('iteration %3d\n', epoch);
        
        % beta1 = max(param.beta1-0.05*(epoch-1),0.1);
        % beta2 = max(param.beta2-0.05*(epoch-1),0.1);
        % beta3 = max(param.beta3-0.05*(epoch-1),0.1);
        
        Sc1 = randperm(n, n_sel);
        
        % update G by groups
        for gi = 1:ngroup
            n_m = length(id_cl{1,gi});
            Gm = G_opt(id_cl{1,gi},:);
            Bm = B(id_cl{1,gi},:);
            Vm = V(id_cl{1,gi},:);
            LTrain_m = LTrain(id_cl{1,gi},:);
            SX1 = LTrain_m * LTrain(Sc1, :)' > 0;
            Sc2 = randperm(n_m, floor(n_sel/ngroup/2)*2); %Sc2 = randperm(n_m, floor(n_sel/ngroup));
            SX2 = LTrain_m * LTrain_m(Sc2, :)' > 0;
            
            Gm = updateColumnG(Gm,B,Bm,Vm,SX1,SX2,Sc1,Sc2,Yg,LTrain_m,nbits,alpha1,alpha2,lambda,beta1,beta2,beta3,n_sel);

            G_opt(id_cl{1,gi},:) = Gm;
        end

        % update B by groups
        for gi = 1:ngroup
            n_m = length(id_cl{1,gi});
            Gm = G_opt(id_cl{1,gi},:);
            Bm = B(id_cl{1,gi},:);
            LTrain_m = LTrain(id_cl{1,gi},:);
            SX1 = LTrain_m * LTrain(Sc1, :)' > 0;
            Sc2 = randperm(n_m, floor(n_sel/ngroup/2)*2); %Sc2 = randperm(n_m, floor(n_sel/ngroup));
            SX2 = LTrain_m * LTrain_m(Sc2, :)' > 0;
            
            Bm = updateColumnB(Bm,G_opt,Gm,SX1,SX2,Sc1,Sc2,nbits,lambda,alpha1,alpha2,n_sel);

            B(id_cl{1,gi},:) = Bm;
        end
    end
end

function Gm = updateColumnG(Gm,B,Bm,Vm,SX1,SX2,Sc1,Sc2,Yg,LTrain_m,nbits,lambda,alpha1,alpha2,beta1,beta2,beta3,n_sel)
    m1 = n_sel; m2 = size(SX2,2);
    n_m = size(Gm,1);
    for k = 1: nbits
        TX1 = lambda * Gm * B(Sc1, :)' / nbits;
        AX1 = 1 ./ (1 + exp(-TX1));
        Bjk = B(Sc1, k)';
        TX2 = lambda * Gm * Bm(Sc2, :)' / nbits;
        AX2 = 1 ./ (1 + exp(-TX2));
        Bmjk = Bm(Sc2, k)';
        tmp = alpha1*lambda*((SX1-AX1).*repmat(Bjk,n_m,1))*ones(m1,1)/nbits...
            +alpha2*lambda*((SX2-AX2).*repmat(Bmjk,n_m,1))*ones(m2,1)/nbits...
            -2*beta1*(Gm(:,k)-LTrain_m*Yg(:,k))-2*beta2*(Gm(:,k)-Vm(:,k))-2*beta3*(Gm(:,k)-Bm(:,k))...
            + (m1*alpha1+m2*alpha2)*lambda^2*Gm(:,k)/(4*nbits^2)+2*(beta1+beta2+beta3)*Gm(:,k);
        Gm_opt = ones(n_m, 1);
        Gm_opt(tmp < 0) = -1;
        Gm(:, k) = Gm_opt;
    end
end


function Bm = updateColumnB(Bm,G,Gm,SX1,SX2,Sc1,Sc2,nbits,lambda,alpha1,alpha2,n_sel)
    m1 = n_sel; m2 = size(SX2,2);
    n_m = size(Bm, 1);
    for k = 1: nbits
        TX1 = lambda * G(Sc1, :) * Bm' / nbits;
        AX1 = 1 ./ (1 + exp(-TX1));
        Gjk = G(Sc1, k)';  %1*8
        TX2 = lambda * Gm(Sc2, :) * Bm' / nbits;
        AX2 = 1 ./ (1 + exp(-TX2));
        Gmjk = Gm(Sc2, k)';  %1*8
        tmp = alpha1*lambda*((SX1-AX1').*repmat(Gjk,n_m,1))*ones(m1,1)/nbits...
            +alpha2*lambda*((SX2-AX2').*repmat(Gmjk,n_m,1))*ones(m2,1)/nbits...
            +(m1*alpha1+m2*alpha2)*lambda^2*Bm(:,k)/(4*nbits^2);
        Bm_opt = ones(n_m, 1);
        Bm_opt(tmp < 0) = -1;
        Bm(:, k) = Bm_opt;
    end
end