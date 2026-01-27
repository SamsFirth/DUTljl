#pragma once
#include <vector>

struct SFT_MoEForwardCache {
    std::vector<std::vector<float>> gate_u; 
    std::vector<std::vector<float>> up_v; 
    void init(int k, int inter_size) {
       if (k > (int)gate_u.size()) {
            gate_u.resize(k);
            up_v  .resize(k);
        }

        for (int i = 0; i < k; ++i) {
            if ((int)gate_u[i].capacity() < inter_size)
                gate_u[i].reserve(inter_size);
            if ((int)up_v[i].capacity()   < inter_size)
                up_v[i].reserve(inter_size);

            gate_u[i].resize(inter_size);
            up_v[i]  .resize(inter_size);

        }
	}
};