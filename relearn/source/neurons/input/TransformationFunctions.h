#pragma once

#include "Types.h"

#include "iostream"

enum class TransformationFunctionType : char {
    Identity,
    Linear
};

/**
 * @brief Pretty-prints the transformation function type to the chosen stream
 * @param out The stream to which to print the transformation function
 * @param element_type The transformation function type to print
 * @return The argument out, now altered with the transformation function type
 */
inline std::ostream& operator<<(std::ostream& out, const TransformationFunctionType& transformation_function_type) {
    if (transformation_function_type == TransformationFunctionType::Identity) {
        return out << "Identity";
    }

    if (transformation_function_type == TransformationFunctionType::Linear) {
        return out << "Linear";
    }

    return out;
}
template <>
struct fmt::formatter<TransformationFunctionType> : ostream_formatter { };

/**
 * Converts a value based on the current step. For example: Decrease the background activity each step
 */
class TransformationFunction {
public:
    virtual ~TransformationFunction() = default;

    TransformationFunction() = default;

    /**
     * Transforms a value based on the current step
     * @param step The current step
     * @param value The value that shall be transformed
     * @return The transformed value
     */
    [[nodiscard]] virtual double transform(RelearnTypes::step_type step, double value) = 0;
    virtual std::unique_ptr<TransformationFunction> clone() = 0;
};

/**
 * Applies no transformation to the value
 */
class IdentityTransformation : public TransformationFunction {
public:
    double transform([[maybe_unused]] const RelearnTypes::step_type step, const double value) override {
        return value;
    }

    std::unique_ptr<TransformationFunction> clone() override {
        return std::make_unique<IdentityTransformation>();
    }
};

/**
 * Decreases the values linear with time
 */
class LinearTransformation : public TransformationFunction {
public:
    /**
     * Constructs new linear transformation object. The activity is multiplied with a linear changing factor_change
     * @param factor_change The change of the factor_change in each step. Negative if decreasing
     * @param factor_start The start value of the factor_change
     * @param factor_cutoff The factor_change will be set to factor_cutoff if it is below (negative factor_change change) or higher (positive factor_change change) the factor_cutoff
     */
    LinearTransformation(double factor_change, double factor_start, double factor_cutoff)
        : factor(factor_change)
        , factor_start(factor_start)
        , factor_cutoff(factor_cutoff) {
    }

    double transform([[maybe_unused]] const RelearnTypes::step_type step, const double value) override {
        const auto m = factor * step + factor_start;
        double cut_off{ 0 };
        if (factor > 0.0) {
            cut_off = fmin(m, factor_cutoff);
        } else {
            cut_off = fmax(m, factor_cutoff);
        }
        auto transformed = value * cut_off;
        return transformed;
    }

    std::unique_ptr<TransformationFunction> clone() override {
        return std::make_unique<IdentityTransformation>();
    }

    static constexpr double default_factor_start{ 1.0 };
    static constexpr double default_factor_cutoff{ 0.0 };
    static constexpr double default_factor{ -0.001 };

private:
    double factor;
    double factor_start;
    double factor_cutoff;
};