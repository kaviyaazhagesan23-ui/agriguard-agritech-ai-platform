# Phase 7.5 — Local Blockchain Verification / Tamper-Evident Audit Layer

## 1. Purpose

PaddyWise AI now has a small local audit ledger for selected farmer decision/listing records. The blockchain provides a tamper-evident commitment after a record is recorded.

It is **not** the forecasting model and does not replace Phase 7's Sell vs Wait calculations.

## 2. Why blockchain is used

The local chain provides a simple append-only sequence of cryptographic commitments. An application record is canonicalized and hashed with SHA-256. Only the hash and minimal block metadata are committed to the chain.

## 3. Architecture

```text
Application Decision Record
        |
        v
Canonical JSON
        |
        v
SHA-256 record hash
        |
        v
Local PaddyWise Block
        |
        v
previous_hash -> block_hash chain
        |
        v
Validation / Verification
        |
        +--> Verified
        +--> Record modified
        +--> Blockchain corruption
        +--> Missing record
```

## 4. Off-chain vs on-chain

### Off-chain

The full decision record remains in application memory/storage. The demonstration record contains only fields already represented by the Phase 7 decision engine plus optional forecast horizon metadata. No farmer identity, phone number, email, password, bank account, or other PII is included.

### On the local blockchain

Each block stores:

- block index
- timestamp
- record ID
- SHA-256 record hash
- previous block hash
- deterministic nonce field (`0`; no mining)
- block hash

The chain is persisted at:

`data/blockchain/paddywise_blockchain.json`

## 5. SHA-256 methodology

Records are serialized as canonical JSON with:

- sorted keys
- compact separators
- UTF-8 encoding
- no random serialization fields
- NaN/Infinity rejected

Therefore the same logical record produces the same SHA-256 digest regardless of dictionary key insertion order.

## 6. Block structure

```text
Block
├── index
├── timestamp
├── record_id
├── record_hash
├── previous_hash
├── nonce = 0
└── block_hash
```

The block hash covers all block fields except `block_hash` itself. There is no proof-of-work and no mining.

## 7. Verification process

`verify_record(record_id, current_record)`:

1. Locate the committed block by record ID.
2. Validate the entire local chain.
3. Recreate the canonical JSON for the current off-chain record.
4. Calculate its SHA-256 hash.
5. Compare that hash with the committed `record_hash`.
6. Return a structured verification result.

A valid chain with a changed application record produces `record_modified`.
A changed block produces `blockchain_corruption`.
An absent record produces `missing_record`.

## 8. Tampering demonstration

The demonstration commits `PW-DEMO-001`, verifies the original record, changes the forecast price in memory, and verifies the modified record again.

Expected conceptual result:

```text
Original record verification: PASS
Modified record verification: FAIL
Blockchain validation: PASS
```

The final `PASS` for blockchain validation is expected because the demonstration changes only the off-chain copy. The committed blockchain itself has not been changed.

## 9. Security limitations

This is a local integrity/audit mechanism, not a distributed blockchain network. Anyone who has write access to the JSON blockchain file can potentially replace the whole file. There is no consensus network, external notarization, proof-of-work, or hardware-backed key protection.

The blockchain proves that a later record matches a previously committed hash; it does not prove that the original application data was truthful.

> **Blockchain provides tamper evidence for records after commitment. It does not independently verify that the original data entered into the system was truthful.**

## 10. Privacy considerations

Only minimal verification metadata is committed. Full application records remain off-chain. The Phase 7 integration used here does not add personal identifiers.

## 11. Why this is a local audit ledger rather than a cryptocurrency system

There are no:

- cryptocurrencies
- tokens
- wallets
- mining
- proof-of-work
- smart contracts
- financial transactions
- external blockchain connections

The `nonce` field is retained only as a deterministic block field; it is always `0` and has no mining function.

## Public API

The reusable integration points are:

- `create_decision_record(...)`
- `commit_record(...)`
- `verify_record(...)`
- `validate_blockchain(...)`
- `load_or_create_blockchain(...)`

These functions are intended for later Streamlit integration without rebuilding the dashboard in Phase 7.5.
