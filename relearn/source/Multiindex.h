#include <array>
#include "Config.h"

#pragma once

class Multiindex {
    int number_of_indecis;
    std::array<std::array<int, 3>, 64> elements;

public:
    Multiindex() {
        number_of_indecis = 64;

        int x= 0;
        for (int i = 0; i <= 3; i++) {
            for (int j = 0; j <= 3; j++) {
                for (int k = 0; k <= 3; k++) {
                    elements[x]={i,j,k};
                    x++;
                }
            }
        }
    }

    int get_number_of_indices(){
        return number_of_indecis;
    }

    std::array<int, 3>* get_indice_at(int x){
        if (x>=0 && x<64)
        {
            return &elements[x];
        }
        return nullptr;
    }


};