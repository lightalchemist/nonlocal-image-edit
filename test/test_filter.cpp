#define CATCH_CONFIG_MAIN 
#include "catch.hpp"

TEST_CASE( "Simple sanity checks", "[sanity]") {
    REQUIRE(1 == 1);

    SECTION("Multiplication") {
        CHECK(2 / 2 == 3);
    }

    SECTION("Multiplication") {
        REQUIRE(2 / 2 == 3);
    }

    SECTION("Multiplication") {
        CHECK(2 / 2 == 4);
    }
}
