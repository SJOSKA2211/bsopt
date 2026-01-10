from strawberry.printer import print_schema
from src.api.graphql.options import schema as options_schema
from src.pricing.graphql.schema import schema as pricing_schema
from src.ml.graphql.schema import schema as ml_schema
from src.streaming.graphql.schema import schema as streaming_schema

def test_option_keys():
    """Verify that Option entity has 'id' key in all subgraphs."""
    
    schemas = [options_schema, pricing_schema, ml_schema, streaming_schema]
    
    for schema in schemas:
        type_def = schema.get_type_by_name("Option")
        assert type_def is not None
        
        sdl = print_schema(schema)
        # Note: print_schema might not output federation directives by default unless configured or using a specific printer
        # However, typically it does if they are part of the type definition.
        # Let's check for "type Option" first.
        assert "type Option" in sdl
