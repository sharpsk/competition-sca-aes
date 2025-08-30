from Crypto.Cipher import AES

pt = bytes.fromhex("07D3580C7695EC206ACB7476E361190C")
ct_expected = bytes.fromhex("6BB092EF6950C6B0117FE63E7FAA5C6D")
k0 = bytes.fromhex("7a75af94ed8e53f03f2233ee97c29ef8")

cipher = AES.new(k0, AES.MODE_ECB)
ct_computed = cipher.encrypt(pt)

print("Expected CT :", ct_expected.hex().upper())
print("Computed CT :", ct_computed.hex().upper())
print("Match?      :", ct_computed == ct_expected)
