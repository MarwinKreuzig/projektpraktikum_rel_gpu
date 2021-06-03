#include <array>
#include "../Config.h"

#pragma once

class Multiindex {
    int number_of_indecis;
    std::array<std::array<unsigned int, 3>, Constants::p*Constants::p*Constants::p> elements;

public:
    Multiindex() {
        number_of_indecis = pow(Constants::p, 3);

        int x= 0;
        for (unsigned int i = 0; i <= 3; i++) {
            for (unsigned int j = 0; j <= 3; j++) {
                for (unsigned int k = 0; k <= 3; k++) {
                    elements[x]={i,j,k};
                    x++;
                }
            }
        }
    }

    int get_number_of_indices(){
        return number_of_indecis;
    }

   const std::array<unsigned int, 3> &get_index(unsigned int x){
        if (x < pow(Constants::p,3))
        {
            return elements.at(x);
        }
    }
};
