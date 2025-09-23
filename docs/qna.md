# Clarity Q&A: Your Project's AI Brain in Slack

This document provides a comprehensive overview of Clarity, an intelligent AI assistant for Slack designed to eliminate project chaos, automate tedious tasks, and provide instant answers.

---

## **Section 1: Solving Project Management Chaos (The "Why")**

### **Q: My team is drowning in Slack messages, meeting notes, and documents. How does Clarity solve this information overload?**

**A:** Clarity is designed to be your project's single source of truth. Instead of you manually digging through information, Clarity does the work. It securely listens to conversations, reads documents (like PDFs and specs), and even transcribes meetings. All this information is organized into an intelligent knowledge base. So, instead of searching, you can simply **ask**. For example, ask "What was the final decision on the Q3 budget?" and get an instant, accurate answer with sources, saving you hours of searching.

### **Q: We constantly have to switch between Slack and Jira. How does Clarity fix this broken workflow?**

**A:** Clarity eliminates the context-switching that kills productivity. It acts as an intelligent bridge between your conversations and your project management tools. When a bug is reported or a new task is mentioned in Slack, Clarity identifies it, drafts a Jira ticket pre-filled with all the relevant context from the conversation, and presents it to you for approval—all without leaving Slack. This can **boost your team's task creation workflow by up to 80%**, turning conversations into actionable tickets in seconds, not minutes.

### **Q: Team alignment is a constant struggle, especially with remote members. How does Clarity help?**

**A:** Clarity acts as a central brain that ensures everyone is on the same page, regardless of their time zone. By creating a reliable, queryable knowledge base from your project's actual conversations and documents, it removes ambiguity. A developer in Istanbul can get the same precise answer about a feature implementation at 3 AM as a project manager in New York got at 3 PM the previous day. This reduces the need for constant check-in meetings and empowers team members to self-serve information, ensuring consistent alignment.

---

## **Section 2: The Power of a Slack AI Assistant (The "What")**

### **Q: What exactly is a "Slack AI Assistant"? How is it different from a standard chatbot?**

**A:** A standard chatbot gives you general answers from the internet. Clarity is a **specialist AI assistant** that learns the unique context of *your* project. It uses a powerful technique called Retrieval-Augmented Generation (RAG), which means it doesn't guess. It first **retrieves** the actual facts from your project's documents, conversations, and meeting transcripts, and then **generates** an answer based only on that verified information. This prevents the "hallucinations" or incorrect answers common with general-purpose AI, ensuring the information is grounded in your project's reality.

### **Q: Can Clarity understand technical conversations and complex documents?**

**A:** Absolutely. Clarity is built to understand the nuances of project management and software development. It processes technical specifications, design documents, and even code snippets shared in conversations. Its Natural Language Processing (NLP) models are designed to identify key entities like feature names, bug reports, deadlines, and stakeholders, allowing it to provide contextually relevant answers and create highly detailed Jira tickets.

### **Q: How does Clarity handle meetings and video calls?**

**A:** You can upload any audio or video recording to a channel with Clarity. It will automatically transcribe the entire conversation with high accuracy, generate a concise summary of the key decisions and action items, and then make the entire transcript searchable. This means you can later ask, "What did Sarah say about the API integration during Tuesday's sync?" and get an exact quote and context.

---

## **Section 3: Feature Deep Dive (The "How")**

### **Q: How does the automated Jira ticket creation work?**

**A:** It's a three-step process designed for speed and accuracy:
1.  **Listen & Understand:** Clarity's AI identifies a potential task from a conversation (e.g., "Kevin found a bug in the checkout flow").
2.  **Draft & Propose:** It automatically drafts a Jira ticket, intelligently suggesting the project, issue type, priority, and description based on the context it has learned. This draft is sent to you in a private Slack message.
3.  **Approve & Create:** You can approve the ticket with a single click, edit any field directly in Slack if needed, or discard it. Once approved, the ticket is created in Jira, and a confirmation with a link is posted back in Slack.

### **Q: What kind of questions can I ask Clarity?**

**A:** You can ask almost anything about your project. Here are some examples:
*   **Status Updates:** "What's the status of the user dashboard redesign?"
*   **Technical Details:** "How is the subscription logic implemented according to the spec?"
*   **Historical Decisions:** "What was the last decision we made about the newsletter?"
*   **Finding Information:** "Where can I find the final design mockups for the mobile app?"
*   **Action Items:** "What are the outstanding action items from yesterday's stand-up?"

### **Q: What tools and file types does Clarity support?**

**A:** Clarity's deepest integration at launch is with **Jira**, allowing for seamless, context-aware ticket creation. We are actively developing integrations for other essential platforms like **Confluence, Asana, Trello, and GitHub** to ensure Clarity fits into your entire workflow. For document analysis, it supports PDF, DOCX, TXT, XLSX, and CSV files, and for meeting intelligence, it can transcribe MP3 and MP4 audio/video files.

---

## **Section 4: Pricing & Early Access**

### **Q: What is the "Founding Member" offer?**

**A:** The Founding Member offer is a one-time opportunity for 200 early adopters to get lifetime premium access to Clarity at a significant discount. For a single payment of $299, you receive a 12-month subscription that begins on the official launch day. This offer includes a permanent 70% discount on all future renewals, priority access to new features, and a direct line to our development team to help shape the product roadmap. You can secure your spot on our website's [Founders' Circle section](#founders-circle). If you're not ready to purchase, you can [join the waitlist](#waitlist) for launch updates.

### **Q: When will Clarity officially launch?**

**A:** We are on track for a public launch in early 2026. Our current focus is on refining the core features and ensuring the platform is robust, secure, and delivers an exceptional user experience. Founding Members will receive priority access before the general release.

---

## **Section 5: Trust & Security**

### **Q: Is my project and conversation data secure?**

**A:** Yes, security is our highest priority. Clarity is built with enterprise-grade security standards. Your data is encrypted both in transit and at rest. Each workspace's knowledge base is completely isolated using Supabase's Row Level Security, meaning no other company or workspace can ever access your data.

### **Q: Who can interact with the bot and access our project's knowledge base?**

**A:** You have full control. A workspace administrator must explicitly grant access to individual users. The bot will learn from all channels it's invited to, but it will only respond to or create tickets for these authorized users, and only in private messages or designated channels. This ensures your project's sensitive information remains confidential.
