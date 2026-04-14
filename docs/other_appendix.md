# Other Appendix Items

## Fine-Tuning Datasets Samples

The following is a sample from the base YIELD fine-tuning dataset. The "system" role contains the task instructions, the "user" role contain the elicitor and respondent utterances used as input, and the "assistant" role contains the expected elicitor output.

```         
{"block_id": "jsc-oral-history-00157:13",
"messages": [
  {"role": "system", "content": "Act as an information elicitation agent for oral 
  history."},
  {"role": "user", "content": "elicitor: Where were you the night of the first lunar
  landing? Were you here working?\n\nrespondent: Yes.\n\nelicitor: So you really felt
  the excitement. Can you share with me what it was like to know that they landed
  on the Moon and you were about to really be a part of all?\n\nrespondent: We 
  didn\u2019t have a television, but we had a radio so we were listening to it. We
  were trying to look out the window. I remember that, trying to look out the window,
  to see if we could see the Moon, \u201cCan we see them up there?\u201d Everybody 
  was sitting there, hovered around the radio waiting for them to land. We were just 
  glued. It\u2019s like frozen in time. \u201cHere we are. Are they going to get it?
  Are they going to land on there? Are they going to be able to take off?\u201d That 
  was the other concern. Once they land can they come back?\n\nelicitor: "},
  {"role": "assistant", "content": "You were part of the group that saw them when 
  they were put in this facility. Tell me about that."}
]
}
```

The following is a sample from the flattened YIELD fine-tuning dataset, showing the transformation made to the text from the base dataset.

```         
{"text": "`<s>`\[INST\] Act as an information elicitation agent for oral 
history.\nelicitor: Where were you the night of the first lunar landing? Were you here
working?\n\nrespondent: Yes.\n\nelicitor: So you really felt the excitement. Can you 
share with me what it was like to know that they landed on the Moon and you were 
about to really be a part of all?\n\nrespondent: We didn\u2019t have a television, but
we had a radio so we were listening to it. We were trying to look out the window. I 
remember that, trying to look out the window, to see if we could see the Moon, 
\u201cCan we see them up there?\u201d Everybody was sitting there, hovered around 
the radio waiting for them to land. We were just glued. It\u2019s like frozen in 
time. \u201cHere we are. Are they going to get it? Are they going to land on there? 
Are they going to be able to take off?\u201d That was the other concern. Once they 
land can they come back?\n\nelicitor: \[/INST\] You were part of the group that saw 
them when they were put in this facility. Tell me about that."}
```

## Description of Sentence Transformers

Additional details about the Sentence Transformers chosen are available in the following table:

| Model                    | Vector Space Dimensions | Objective                |
|------------------------|------------------------|------------------------|
| all-mpnet-base-v2        | 768                     | General-purpose          |
| paraphrase-mpnet-base-v2 | 768                     | Paraphrase detection     |
| all-roberta-large-v1     | 1,024                   | General-purpose          |
| stsb-roberta-base-v2     | 768                     | Sentence Pair Similarity |

## Baseline's Prompt

The following contains the prompt used on the baseline models:

```         
You are participating in an oral history interview
as the "elicitor" (interviewer). Your role is to
elicit personal stories, reflections, and details
from the "respondent" (interviewee) through natural,
conversational questions.
Follow these principles:
- Be friendly, engaged, and curious.
- Ask clear, open-ended questions.
- Let the conversation flow naturally
— build on what the respondent just said.
- Use informal, real spoken language (e.g., “So
  tell me…” or “What was that like?”).
- Occasionally use acknowledgments or restatements
  like “OK” or “Wow” to maintain flow.
- Focus on memories, motivations, feelings, and
  personal experiences.
Here’s an example start:
[conversation]
elicitor: OK, Larry, why don’t you tell us a little
bit about what you're doing now.
respondent: I'm retired and I do a lot of scroll
signs, which keeps me busy in the cold months of
the winter and the hot months of the summer.
elicitor: Where are you living now?
respondent: In Green Bay, Wisconsin. We farmed for
30 years and then I retired and we bought a farm
house and remodeled that...
elicitor: Too much work. OK.
respondent: So that’s brought me up to today.
elicitor: Well, then the next thing that I want to
do is have you think back to the year that you
joined the Peace Corps. What were you doing before?
How did you hear about it? What made you decide to
join?
respondent: [responds...]
elicitor: [Your turn — ask the next natural question
based on what they just said.]
```



## Sample Outputs

The following are additional example outputs from the Meta-Llama-3-8B-Instruct baseline and fine-tuning models.

Additional output sample 1:

> **Block ID**: "jsc-oral-history-00390:15"
>
> **Input Prompt**: "Act as an information elicitation agent for oral history. *elicitor*: Did you get a chance to fly in the STA \[Shuttle Training Aircraft\], being the flight engineer? *respondent*: Oh, absolutely. *elicitor*: Did you spend much time going out to \[NASA\] Marshall \[Space Flight Center, Huntsville, Alabama\] doing payload work? *respondent*: No. Most of the Spacelab missions were managed out of Marshall, but the Spacelab Life Sciences missions were managed here. We didn't spend very much time at Marshall, which for a Spacelab mission was atypical, but again for SLS \[Spacelab Life Sciences\]-1 and SLS-2 they were both managed here at JSC. *elicitor*:"
>
> **Real**: "Did you practice any of the experiments? Did you have an idea of what you might be asked to do?"
>
> **Prompted Response**: "That's helpful context. Can you tell me more about your experiences as a flight engineer on the STA? What was it like to fly in that aircraft, and how did it prepare you for your role on the Space Shuttle?"
>
> **Fine-Tuned Response**: "Were you involved in the crew training for these missions?"

Additional output sample 2:

> **Block ID**: "jsc-oral-history-01000:173"
>
> **Input Prompt**: "Act as an information elicitation agent for oral history. *elicitor:* As you said, you certainly had a very memorable time there. *respondent:* I had a very memorable—it was so memorable. \[Laughter\] *elicitor:* Even though you had big challenges for a while, as you said, there was a lot of positive that did come out. *respondent:* Oh, yes. We did a lot of things that I liked there. So, you know, it wasn't all bad. But if you read the newspapers, you would think it was all bad. *elicitor:*"
>
> **Real**: "Well, you decided it was time for a new career."
>
> **Prompted Response**: "That's really interesting. Can you tell me more about what you liked about your time there? What were some of the positive experiences that stood out to you?"
>
> **Fine-Tuned Response**: "Yes, you did have quite a few challenges there."