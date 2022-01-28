
# from decimal import Decimal


# def test_empty__return_empty():
#   result = convert_weights_to_percent_inplace({})
#   assert result == {}


# def test_1_1_1__returns_one_third_for_all():
#   weights = dict.fromkeys(range(3), 1)

#   convert_weights_to_percent_inplace(weights)

#   assert weights == {
#     0: Decimal(1) / 3,
#     1: Decimal(1) / 3,
#     2: Decimal(1) / 3,
#   } 

#   assert sum(weights.values()) == 1
