function V = generate_comspace(XTrain,YTrain,LTrain,param)
    
    max_iter = param.max_iter_V;
    etax = param.etax; etay = param.etay; etal = param.etal;
    theta = param.theta;
    dV = param.dV;
    
    n = size(XTrain,1);
    dX = size(XTrain,2);
    dY = size(YTrain,2);
    dL = size(LTrain,2);
    
    V = randn(n,dV);
    for iter = 1:max_iter
        % U-step
        Ux = (XTrain'*XTrain*etax+theta*eye(dX)) \ (XTrain'*V*etax);
        Uy = (YTrain'*YTrain*etay+theta*eye(dY)) \ (YTrain'*V*etay);
        Ul = (LTrain'*LTrain*etal+theta*eye(dL)) \ (LTrain'*V*etal);
        
        % V-step
        V = XTrain*Ux*etax+YTrain*Uy*etay+LTrain*Ul*etal;
    end
    
end