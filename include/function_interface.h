//
// Created by Richard Lettich on 4/13/21.
//

#ifndef ACSPGEMM_FUNCTION_INTERFACE_H
#define ACSPGEMM_FUNCTION_INTERFACE_H

#endif //ACSPGEMM_FUNCTION_INTERFACE_H


struct CanonicalImplementation {};


/// We wish to define an bounded interface for C++ for a generalized semiring over two Types,
/// T,U  to describe two functions
///  multiply : T x T -> U
///       add : U x U -> U
///
/// We do this with an additional type called TypeLabel.
/// The purpose is to act as a type-level discriminant to allow multiple semiring over the same two types.
/// e.g. we can create a semiring for regular addition / multiplication over doubles
///
/// struct Regular{};
///
///     Monoid<double, double, Regular> { ... impl...  }
///
/// and also one for a  tropical semiring....
///
///     Monoid<double, double, Tropical> { ... impl ... }
///
/// without conflict.

template <typename T, typename U, typename TypeLabel>
SemiRing {
    U multiply(T& a, T& b);
    U add(U& a, U& b);

    T MultiplicativeIdentity();
    U AdditiveIdentity();
};






