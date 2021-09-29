.. role:: hidden
    :class: hidden-section

Compare parsed addresses
********************************

.. code-block:: python


First, let's download the trained parser to compare the parsing of the address

    from deepparse.parser import AddressParser
    address_parser = AddressParser(model_type="bpemb")

Then let's create an AddressesComparer object compare two already parsed addresses

.. code-block:: python

    addresses_comparer = AddressesComparer(address_parser)

    first_tags_comparison = [("350", "StreetNumber"), ("rue des Lilas", "StreetName"),
                                  ("Ouest Québec", "Municipality"), ("Québec", "Province"), ("G1L 1B6", "PostalCode")]


    address_parser = AddressParser(model_type="bpemb")
    addresses_comparer = AddressesComparer(address_parser)

    first_parsing_comparison = addresses_comparer.compare_tags(first_tags_comparison)

 Now print the comparison report so see if the source parsing is the same as our parser   

.. code-block:: python

    first_parsing_comparison.comparison_report()

It is also possible to compare parsed addresses that contains probabilities
for each Tags

.. code-block:: python
    second_comparison_with_probs =  [
            ('350', ('StreetNumber', 1.0)),
            ('rue', ('StreetName', 0.9987)),
            ('des', ('StreetName', 0.9993)),
            ('Lilas', ('StreetName', 0.8176)),
            ('Ouest', ('Orientation', 0.781)),
            ('Quebec', ('Municipality', 0.9768)),
            ('Quebec', ('Province', 1.0)),
            ('G1L', ('PostalCode', 0.9993)),
            ('1B6', ('PostalCode', 1.0))]

    second_parsing_comparison_with_probs = addresses_comparer.compare_tags(second_comparison_with_probs)

    second_parsing_comparison_with_probs.comparison_report()

Note that if the input has probabilities, the comparison report will also print its
probabilities for each tags. In order to get the probabilities even without passing parsing 
you can specificy with_prob = True when calling the compare_tags method.

.. code-block:: python
    first_parsing_comparison_with_probs = addresses_comparer.compare_tags(first_comparison, with_prob= True)

It is also possible to make multiple comparison with the same call when using a list of parsed addresses.

.. code-block:: python
    multiple_comparisons = addresses_comparer.compare_tags([first_tags_comparison, second_comparison_with_probs])
    multiple_comparisons.comparison_report()
    multiple_comparisons.comparison_report()

By default, if there is a parsing that contains probabilities, all the comparison reports will have probabilities.

It is also possible to compare two raw addresses and check if they are identical, equivalent or not equivalent.
Identical means the parsing is the same for the two addresses and the raw addresses are also equals.
Equivalent means the parsing is the same for the two addresses, but the raw addresses differ. 
Not equivalent means the parsing is different for the two addresses.

Enough talking, let's try it using the same addresses_comparer. Here is an example for an identical comparison,
an equivalent comparison and a different comparison.

.. code-block:: python

    raw_address_original = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"
    raw_address_identical = "350 rue des Lilas Ouest Quebec Quebec G1L 1B6"
    raw_address_equivalent = "350  rue des Lilas Ouest Quebec Quebec G1L 1B6"
    raw_address_diff_streetNumber = "450 rue des Lilas Ouest Quebec Quebec G1L 1B6"

    raw_addresses_multiples_comparisons = addresses_comparer.compare_raw([(raw_address_original,
                                                                            raw_address_identical)
                                                                            ,(raw_address_original,
                                                                            raw_address_equivalent),
                                                                           (raw_address_original,
                                                                            raw_address_diff_streetNumber)],
                                                                            with_prob = True)
    raw_addresses_multiples_comparisons[0].comparison_report()
    raw_addresses_multiples_comparisons[1].comparison_report()
    raw_addresses_multiples_comparisons[2].comparison_report()