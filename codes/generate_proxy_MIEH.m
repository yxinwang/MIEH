function Yg = generate_proxy(LTrain,param)
    
    max_iter = param.max_iter_proxy;
    mu = param.mu;
    nbits = param.nbits;
    
    c = size(LTrain,2);
    
    Cg = LTrain'*LTrain;
    ttp = diag(Cg);
    Cg = Cg./(repmat(ttp,1,c)+repmat(ttp',c,1)-Cg);
    Cg = roundn(Cg,-4);
    Cg(isnan(Cg)) = 1;
	
    
    % Hadamard Yg
    %Yg = sign(randn(c,nbits));%=====
    if c==10
        Yg = hadamard(max(nbits,12));
    elseif c==255
        Yg = hadamard(256);
    else
        Yg = hadamard(max(nbits,24));
    end
    Yg = Yg(1:c,1:nbits);
    for iter = 1:max_iter
        Deg = (Cg-mu*ones(c,c)+mu*eye(c));
        Lap = Deg*ones(c,1)-Deg;
        
        %P-step
        F = -Lap*Yg;
        if nbits < c
            Temp = F'*F-1/c*(F'*ones(c,1)*(ones(1,c)*F));
            [~,Lmd,QQ] = svd(Temp); clear Temp
            index = (diag(Lmd)>1e-6);
            Q = QQ(:,index); Q_ = orth(QQ(:,~index));
            P = (F-1/c*ones(c,1)*(ones(1,c)*F)) *  (Q / (sqrt(Lmd(index,index))));
            P_ = orth(randn(c,nbits-length(find(index==1))));
            Pg = sqrt(c)*[P P_]*[Q Q_]';
        else
            [PP,Lmd,QQ] = svd(F,0);
            index = (diag(Lmd)>1e-6);
            Q = QQ(:,index); Q_ = orth(QQ(:,~index)); clear QQ
            P = PP(:,index); P_ = orth(PP(:,~index)); clear PP
            Pg = sqrt(c)*[P P_]*[Q Q_]';
        end
        
        %Y-step
        Yg = sign(-Lap*Pg);
    end
    
end