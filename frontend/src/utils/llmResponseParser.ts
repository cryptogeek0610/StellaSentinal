/**
 * Parser utility for raw LLM response text
 * Extracts structured sections from AI root cause analysis responses
 */

export interface ParsedAnalysis {
  primaryHypothesis: {
    title: string;
    confidence: 'High' | 'Medium' | 'Low';
    description: string;
  };
  supportingEvidence: string[];
  alternativeHypothesis?: {
    title: string;
    confidence: 'High' | 'Medium' | 'Low';
    whyLessLikely: string;
  };
  recommendedActions: {
    urgency: 'Immediate' | 'Soon' | 'Monitor';
    actions: string[];
  };
  businessImpact: string;
}

/**
 * Clean raw LLM text by stripping model-specific artifacts
 */
function cleanLLMText(text: string): string {
  return text
    // Strip box markers (used by some models)
    .replace(/<\|begin_of_box\|>/g, '')
    .replace(/<\|end_of_box\|>/g, '')
    // Strip thinking tags (used by DeepSeek R1 and similar)
    .replace(/<think>[\s\S]*?<\/think>/g, '')
    // Strip other common artifacts
    .replace(/<\|.*?\|>/g, '')
    // Clean up excessive whitespace
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

/**
 * Extract text between two section headers
 */
function extractSection(
  text: string,
  startHeader: string,
  endHeaders: string[]
): string {
  const startRegex = new RegExp(`${startHeader}[:\\s]*`, 'i');
  const startMatch = text.match(startRegex);

  if (!startMatch) return '';

  const startIndex = startMatch.index! + startMatch[0].length;
  let endIndex = text.length;

  for (const endHeader of endHeaders) {
    const endRegex = new RegExp(`\\n\\s*${endHeader}[:\\s]*`, 'i');
    const endMatch = text.slice(startIndex).match(endRegex);
    if (endMatch && endMatch.index !== undefined) {
      const potentialEnd = startIndex + endMatch.index;
      if (potentialEnd < endIndex) {
        endIndex = potentialEnd;
      }
    }
  }

  return text.slice(startIndex, endIndex).trim();
}

/**
 * Parse a field value from text like "Title: Some Value"
 */
function parseField(text: string, fieldName: string): string {
  const regex = new RegExp(`${fieldName}[:\\s]+([^\\n]+)`, 'i');
  const match = text.match(regex);
  return match ? match[1].trim() : '';
}

/**
 * Parse confidence level from various formats
 */
function parseConfidence(text: string): 'High' | 'Medium' | 'Low' {
  const confidenceText = parseField(text, 'Confidence') || text;
  const lower = confidenceText.toLowerCase();

  if (lower.includes('high')) return 'High';
  if (lower.includes('medium')) return 'Medium';
  if (lower.includes('low')) return 'Low';

  // Try to parse percentage-based confidence
  const percentMatch = confidenceText.match(/(\d+)\s*%/);
  if (percentMatch) {
    const percent = parseInt(percentMatch[1], 10);
    if (percent >= 70) return 'High';
    if (percent >= 40) return 'Medium';
    return 'Low';
  }

  return 'Medium'; // Default
}

/**
 * Parse urgency level from various formats
 */
function parseUrgency(text: string): 'Immediate' | 'Soon' | 'Monitor' {
  const lower = text.toLowerCase();

  if (lower.includes('immediate') || lower.includes('critical') || lower.includes('urgent')) {
    return 'Immediate';
  }
  if (lower.includes('soon') || lower.includes('moderate')) {
    return 'Soon';
  }
  if (lower.includes('monitor') || lower.includes('low')) {
    return 'Monitor';
  }

  return 'Soon'; // Default
}

/**
 * Check if an evidence item mentions a benign-high feature as a problem.
 * High free storage, high memory, strong signal are GOOD - not problems.
 */
function isBenignHighEvidence(item: string): boolean {
  const benignPatterns = [
    // Free Storage mentioned as high/above baseline (benign)
    /Free\s*Storage.*(?:above|higher|increased|significant|large|high)/i,
    /Free\s*Storage.*\d+\.?\d*\s*(?:GB|MB|TB).*(?:above|higher|deviation)/i,
    /(?:High|Large|Significant|Elevated).*[Ff]ree\s*[Ss]torage/i,
    /Free\s*Storage.*(?:\+|positive).*deviation/i,
    /Free\s*Storage.*[5-9]\d*\.?\d*\s*σ/i, // High positive sigma
    // Memory mentioned as high (benign)
    /(?:Free|Available)\s*Memory.*(?:above|higher|high)/i,
    // Battery level high (benign)
    /Battery\s*Level.*(?:high|above|good)/i,
    // Signal strength strong (benign)
    /Signal\s*Strength.*(?:strong|high|good|above)/i,
  ];

  return benignPatterns.some(pattern => pattern.test(item));
}

/**
 * Parse a list of items (bullet points or numbered)
 */
function parseListItems(text: string): string[] {
  const lines = text.split('\n');
  const items: string[] = [];

  for (const line of lines) {
    // Match bullet points (-, *, •) or numbered items (1., 1), etc.)
    const match = line.match(/^\s*[-*•]?\s*\d*[.):]?\s*(.+)/);
    if (match && match[1].trim()) {
      const item = match[1].trim();
      // Skip if it looks like a section header
      if (!item.match(/^(PRIMARY|ALTERNATIVE|SUPPORTING|RECOMMENDED|BUSINESS|Title|Confidence|Urgency)/i)) {
        // Skip benign-high evidence (high free storage is good, not a problem)
        if (!isBenignHighEvidence(item)) {
          items.push(item);
        }
      }
    }
  }

  return items;
}

/**
 * Check if text needs parsing (contains structured LLM output markers)
 */
export function needsParsing(description: string | undefined | null): boolean {
  if (!description) return false;

  return (
    description.includes('PRIMARY HYPOTHESIS') ||
    description.includes('<|begin_of_box|>') ||
    description.includes('SUPPORTING EVIDENCE') ||
    description.includes('RECOMMENDED ACTIONS') ||
    description.includes('BUSINESS IMPACT') ||
    description.includes('ALTERNATIVE HYPOTHESIS')
  );
}

/**
 * Parse raw LLM response text into structured analysis object
 */
export function parseAIResponse(rawText: string): ParsedAnalysis | null {
  const text = cleanLLMText(rawText);

  // If no structured content detected, return null
  if (!text.includes('PRIMARY HYPOTHESIS') && !text.includes('Title:')) {
    return null;
  }

  // Extract each section
  const primarySection = extractSection(
    text,
    'PRIMARY HYPOTHESIS',
    ['SUPPORTING EVIDENCE', 'ALTERNATIVE HYPOTHESIS', 'RECOMMENDED ACTIONS', 'BUSINESS IMPACT']
  );

  const evidenceSection = extractSection(
    text,
    'SUPPORTING EVIDENCE',
    ['ALTERNATIVE HYPOTHESIS', 'RECOMMENDED ACTIONS', 'BUSINESS IMPACT']
  );

  const alternativeSection = extractSection(
    text,
    'ALTERNATIVE HYPOTHESIS',
    ['RECOMMENDED ACTIONS', 'BUSINESS IMPACT']
  );

  const actionsSection = extractSection(
    text,
    'RECOMMENDED ACTIONS',
    ['BUSINESS IMPACT']
  );

  const impactSection = extractSection(
    text,
    'BUSINESS IMPACT',
    []
  );

  // Parse primary hypothesis
  const primaryTitle = parseField(primarySection, 'Title') || 'Root Cause Analysis';
  const primaryConfidence = parseConfidence(primarySection);
  const primaryDescription = parseField(primarySection, 'Description') ||
    primarySection.replace(/Title:[^\n]+\n?/, '').replace(/Confidence:[^\n]+\n?/, '').trim();

  // Parse supporting evidence
  const supportingEvidence = parseListItems(evidenceSection);

  // Parse alternative hypothesis (if present)
  let alternativeHypothesis: ParsedAnalysis['alternativeHypothesis'] | undefined;
  if (alternativeSection) {
    const altTitle = parseField(alternativeSection, 'Title');
    const altConfidence = parseConfidence(alternativeSection);
    const whyLessLikely = parseField(alternativeSection, 'Why less likely') ||
      alternativeSection.replace(/Title:[^\n]+\n?/, '').replace(/Confidence:[^\n]+\n?/, '').trim();

    if (altTitle || whyLessLikely) {
      alternativeHypothesis = {
        title: altTitle || 'Alternative Cause',
        confidence: altConfidence,
        whyLessLikely: whyLessLikely,
      };
    }
  }

  // Parse recommended actions
  const urgency = parseUrgency(actionsSection);
  const actions = parseListItems(actionsSection.replace(/Urgency:[^\n]+\n?/, ''));

  // Parse business impact
  const businessImpact = impactSection || '';

  return {
    primaryHypothesis: {
      title: primaryTitle,
      confidence: primaryConfidence,
      description: primaryDescription,
    },
    supportingEvidence,
    alternativeHypothesis,
    recommendedActions: {
      urgency,
      actions,
    },
    businessImpact,
  };
}
