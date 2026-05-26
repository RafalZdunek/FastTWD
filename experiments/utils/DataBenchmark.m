function [X] = DataBenchmark(bench)

switch bench

    case 1 % CP

        I = [20 20 20 20];
        R = 4;
        
        U = cell([length(I),1]);
        for n = 1:length(I)
            U{n} = max(0,randn([I(n) R]));
        end
        X = cp_full_local(U);

    case 2 % Standard Tucker

        I = [20 20 20 20];
        R = [4 4 4 4];
             
        Core = max(0,randn(R));
        U = cell([length(I),1]);
        for n = 1:length(I)
            U{n} = max(0,randn([I(n) R(n)]));
        end
        X = tucker_full_local(Core, U);

    case 3 % TR

        I = [20 20 20 20];
        R = [4 4 4 4];
             
        U = cell([length(I),1]);
        for n = 1:length(I)
            if n < length(I)
               G{n} = max(0,randn([R(n) I(n) R(n+1)]));
            else
               G{n} = max(0,randn([R(n) I(n) R(1)])); 
            end
        end
        X = squeeze(outer_ring_prod(G)); 

    case 4 % Standard TW

        I = [20 20 20 20];
        R = [4 4 4 4];
        L = [4 4 4 4];
        
        Core = max(0,randn(L));
        G = cell([length(I),1]);
        for n = 1:length(I)
            if n < length(I)
               G{n} = max(0,randn([R(n) I(n) L(n) R(n+1)]));
            else
               G{n} = max(0,randn([R(n) I(n) L(n) R(1)])); 
            end
        end
       X = cores_prod_single_tw(G,Core);
   
end % switch


end

