function X = cores_prod_single_tw(G,C)

    Nc = length(G);
    X = tensorprod(G{1},C,3,1);
    X = permute(X,[4:Nc+2,1:3]);

    for n = 1:Nc-2
        X = tensorprod(X,G{n+1},[1,ndims(X)],[3,1]);
    end
    X = tensorprod(X,G{Nc},[1, 2, ndims(X)],[3, 4, 1]);

end