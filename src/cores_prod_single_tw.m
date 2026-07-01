function X = cores_prod_single_tw(G,C)
%CORES_PROD_SINGLE_TW Reconstruct a tensor from its TW factors and core.
% Explicit logical tensor orders preserve trailing singleton dimensions.

    Nc = length(G);
    logicalNdimsX = Nc + 2;

    X = tensor_contraction_explicit( ...
        G{1}, C, 3, 1, 4, Nc);

    X = permute(X, [4:logicalNdimsX, 1:3]);

    for n = 1:Nc-2
        X = tensor_contraction_explicit( ...
            X, G{n+1}, ...
            [1, logicalNdimsX], [3, 1], ...
            logicalNdimsX, 4);
    end

    X = tensor_contraction_explicit( ...
        X, G{Nc}, ...
        [1, 2, logicalNdimsX], [3, 4, 1], ...
        logicalNdimsX, 4);
end