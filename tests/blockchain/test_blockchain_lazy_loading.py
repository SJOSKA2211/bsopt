"""
Test lazy loading behavior for the src.blockchain package.
"""
import sys


class TestBlockchainLazyLoading:
    def setup_method(self):
        # Clear any cached imports from previous tests
        modules_to_clear = [
            mod for mod in sys.modules.keys() 
            if mod.startswith('src.blockchain') or mod == 'web3'
        ]
        for mod in modules_to_clear:
            del sys.modules[mod]
        # Re-import src.blockchain to ensure a clean state for each test
        if 'src.blockchain' in sys.modules:
            del sys.modules['src.blockchain']

    def test_blockchain_does_not_load_heavy_deps_on_import(self):
        """
        Verify that importing src.blockchain does not load Web3.py.
        """
        assert 'web3' not in sys.modules

    def test_blockchain_loads_dep_on_attribute_access(self):
        """
        Verify that accessing a Web3.py-dependent class in src.blockchain triggers the lazy load.
        """
        import src.blockchain
        
        # Accessing DeFiOptionsProtocol should trigger import
        _ = src.blockchain.DeFiOptionsProtocol
        
        assert 'web3' in sys.modules
        
    def test_dir_returns_all_exports(self):
        """
        Verify dir() returns all exported names.
        """
        import src.blockchain
        exports = dir(src.blockchain)
        assert 'DeFiOptionsProtocol' in exports
        assert 'ChainlinkOracle' in exports
        assert 'IPFSStorage' in exports
        assert 'OptionsContractDeployer' in exports
        # Should not include private members
        assert '_import_map' not in exports