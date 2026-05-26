function G_ring = outer_ring_prod(G)

Nc = length(G);
G_ring = G{1};

for n = 1:Nc-2
    G_ring = tensorprod(G_ring,G{n+1},ndims(G_ring),1);
end
G_ring = tensorprod(G_ring,G{Nc},[ndims(G_ring) 1],[1 ndims(G{Nc})]);

end