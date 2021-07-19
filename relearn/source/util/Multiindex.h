#include <array>
#include "../Config.h"

#pragma once

class Multiindex {
    int number_of_indecis;
    std::array<std::array<unsigned int, 3>, Constants::p3> elements;

public:
    Multiindex() {
        number_of_indecis = Constants::p3;

        int x= 0;
        for (unsigned int i = 0; i < Constants::p; i++) {
            for (unsigned int j = 0; j < Constants::p; j++) {
                for (unsigned int k = 0; k < Constants::p; k++) {
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
