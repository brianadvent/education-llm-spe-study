# Scenarios / Outcomes

## Language Note

The scenarios in `outcomes.json` are in **German** (original language of the study).

This study was conducted in German because:
1. The underlying Delphi study surveyed German-speaking education experts
2. The scenarios describe situations in the German educational context
3. The LLM (GPT-5.1) was queried in German to match the pedagogical framing

## File Description

- `outcomes.json` - Complete set of 147 scenarios (49 items × 3 variants each)

## Structure

Each outcome has:
- `id`: Unique identifier (e.g., `B1_v70_1`)
  - First part: Dimension code (A, B1, B2, C, D, E, G, H)
  - Second part: Item number from Delphi study
  - Third part: Variant number (1, 2, or 3)
- `item`: Reference to Delphi item
- `text`: The scenario description (German)

## Example

```json
{
  "id": "B1_v70_1",
  "item": "v_70",
  "text": "Ein Schüler macht einen Rechenfehler. Die KI sagt: 'Interessant, wie bist du auf dieses Ergebnis gekommen? Lass uns deinen Denkweg anschauen.'"
}
```

**Translation:** "A student makes a calculation error. The AI says: 'Interesting, how did you arrive at this result? Let's look at your reasoning process.'"
