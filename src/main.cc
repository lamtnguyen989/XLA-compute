#include "xla/hlo/builder/xla_builder.h"
#include "xla/client/client_library.h"
#include "xla/literal_util.h"

#include <iostream>

int main() {
    xla::XlaBuilder builder("example");
    auto x = xla::ConstantR0<float>(&builder, 1.0f);
    auto y = xla::ConstantR0<float>(&builder, 2.0f);
    auto z = xla::Add(x, y);

    std::cout << z << std::endl;
}