import { useRef, useState } from "react";

import {
  Bot,
  FileText,
  Send,
  Paperclip,
  User,
  Sparkles,
  X,
  BookOpen,
} from "lucide-react";

import { members } from "../data/mockData";

function RAGAssistant() {

  const [selectedMember, setSelectedMember] =
    useState(members[0]);

  const [question, setQuestion] = useState("");

  const [uploadedFile, setUploadedFile] =
    useState(null);

  const [messages, setMessages] = useState([
    {
      id: 1,
      role: "assistant",
      text:
        "Hello! I can help you understand SDOH risk, intervention priorities, and evidence-based recommendations. Select a member and ask me a question.",
      sources: [],
    },
  ]);

  const fileInputRef = useRef(null);


  const exampleQuestions = [
    "Why is this member considered high risk?",
    "What are the major SDOH factors affecting this member?",
    "Why was transportation assistance recommended?",
    "What intervention should be prioritized first?",
  ];


  function generateMockResponse(text) {

    const lower = text.toLowerCase();

    if (
      lower.includes("transportation")
    ) {
      return {
        text:
          `Transportation Assistance is currently ranked as a high-priority intervention for ${selectedMember.name}. The recommendation is based on elevated transportation-related SDOH risk and its potential effect on healthcare access.`,
        sources: [
          "SDOH Knowledge Base",
          "Member SDOH Profile",
          "Intervention Ranking Model",
        ],
      };
    }

    if (
      lower.includes("risk")
    ) {
      return {
        text:
          `${
  selectedMember.name ||
  selectedMember.memberName ||
  selectedMember.member_id ||
  selectedMember.memberId ||
  selectedMember.id
} has an overall SDOH risk score of ${Math.round(
            selectedMember.riskScore * 100
          )}%. The score reflects multiple social risk factors identified from the member's available SDOH information.`,
        sources: [
          "Member Risk Profile",
          "Risk Prediction Model",
        ],
      };
    }

    if (
      lower.includes("factor") ||
      lower.includes("sdoh")
    ) {
      return {
        text:
          `The major SDOH factors identified for ${
  selectedMember.name ||
  selectedMember.memberName ||
  selectedMember.member_id ||
  selectedMember.memberId ||
  selectedMember.id
} include transportation, food access, housing, economic stability, and healthcare access. The relative contribution depends on the available member-level data.`,
        sources: [
          "Member SDOH Profile",
          "SDOH Knowledge Base",
        ],
      };
    }

    return {
      text:
        `Based on the available information for ${selectedMember.name}, I can analyze SDOH risk factors, intervention priorities, and supporting evidence. This response will be generated from the connected knowledge base and member context once the RAG backend is connected.`,
      sources: [
        "Member Context",
        "SDOH Knowledge Base",
      ],
    };
  }


  function sendMessage(messageText = question) {

    const trimmed = messageText.trim();

    if (!trimmed) {
      return;
    }

    const userMessage = {
      id: Date.now(),
      role: "user",
      text: trimmed,
      sources: [],
    };

    const response =
      generateMockResponse(trimmed);

    const assistantMessage = {
      id: Date.now() + 1,
      role: "assistant",
      text: response.text,
      sources: response.sources,
    };

    setMessages((previous) => [
      ...previous,
      userMessage,
      assistantMessage,
    ]);

    setQuestion("");
  }


  function handleFileChange(event) {

    const file = event.target.files?.[0];

    if (file) {
      setUploadedFile(file);
    }
  }


  return (
    <div className="rag-page">

      {/* Header */}

      <div className="page-header">

        <div>

          <h1>
            SDOH AI Assistant
          </h1>

          <p>
            Ask questions about member risk,
            interventions, and SDOH evidence.
          </p>

        </div>

        <div className="rag-status">

          <Sparkles size={17} />

          RAG Intelligence

        </div>

      </div>


      {/* Member Context */}

      <div className="rag-context">

        <div className="rag-context-title">

          <User size={18} />

          <div>

            <span>
              Current Member
            </span>

            <strong>
                {selectedMember.name ||
                selectedMember.memberName ||
                selectedMember.member_id ||
                selectedMember.memberId ||
                selectedMember.id}
            </strong>

          </div>

        </div>


        <select
          value={selectedMember.id}
          onChange={(event) => {

            const member = members.find(
  (item) =>
    item.id === event.target.value ||
    item.member_id === event.target.value ||
    item.memberId === event.target.value
);

            if (member) {
              setSelectedMember(member);
            }

          }}
        >

          {members.map((member) => {

  const memberLabel =
    member.name ||
    member.memberName ||
    member.member_id ||
    member.memberId ||
    member.id;

  const memberValue =
    member.id ||
    member.member_id ||
    member.memberId;

  return (
    <option
      key={memberValue}
      value={memberValue}
    >
      {memberLabel}
    </option>
  );
})}

        </select>


        <div className="rag-member-risk">

          Risk Score

          <strong>
            {Math.round(
              selectedMember.riskScore * 100
            )}
            %
          </strong>

        </div>

      </div>


      {/* Main Chat */}

      <div className="rag-container">

        {/* Chat */}

        <div className="rag-chat">

          <div className="chat-header">

            <div className="chat-bot-icon">

              <Bot size={20} />

            </div>

            <div>

              <strong>
                SDOH Intelligence Assistant
              </strong>

              <span>
                Personalized analysis
              </span>

            </div>

          </div>


          {/* Messages */}

          <div className="messages">

            {messages.map((message) => (

              <div
                key={message.id}
                className={`message ${
                  message.role
                }`}
              >

                <div className="message-icon">

                  {message.role === "assistant" ? (
                    <Bot size={15} />
                  ) : (
                    <User size={15} />
                  )}

                </div>


                <div className="message-content">

                  <p>
                    {message.text}
                  </p>


                  {message.sources?.length > 0 && (

                    <div className="message-sources">

                      <div className="source-heading">

                        <BookOpen size={13} />

                        Sources

                      </div>

                      {message.sources.map(
                        (source) => (

                          <span key={source}>
                            {source}
                          </span>

                        )
                      )}

                    </div>

                  )}

                </div>

              </div>

            ))}

          </div>


          {/* Suggested Questions */}

          <div className="suggested-questions">

            <span>
              Try asking
            </span>

            <div>

              {exampleQuestions.map(
                (item) => (

                  <button
                    key={item}
                    onClick={() =>
                      sendMessage(item)
                    }
                  >
                    {item}
                  </button>

                )
              )}

            </div>

          </div>


          {/* Uploaded File */}

          {uploadedFile && (

            <div className="uploaded-file">

              <FileText size={17} />

              <div>

                <strong>
                  {uploadedFile.name}
                </strong>

                <span>
                  Ready for RAG processing
                </span>

              </div>

              <button
                onClick={() =>
                  setUploadedFile(null)
                }
              >
                <X size={15} />
              </button>

            </div>

          )}


          {/* Input */}

          <div className="chat-input-area">

            <button
              className="attach-button"
              onClick={() =>
                fileInputRef.current?.click()
              }
            >
              <Paperclip size={18} />
            </button>

            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".pdf,.txt,.doc,.docx"
              hidden
            />


            <input
              value={question}
              onChange={(event) =>
                setQuestion(event.target.value)
              }
              onKeyDown={(event) => {

                if (
                  event.key === "Enter" &&
                  !event.shiftKey
                ) {
                  event.preventDefault();
                  sendMessage();
                }

              }}
              placeholder="Ask about this member's SDOH risk..."
            />


            <button
              className="send-button"
              onClick={() =>
                sendMessage()
              }
            >
              <Send size={18} />
            </button>

          </div>

        </div>


        {/* Right Context Panel */}

        <div className="rag-side-panel">

          <div className="dashboard-card">

            <div className="card-header">

              <div>

                <h2>
                  Member Context
                </h2>

                <p>
                  Information supplied to the
                  assistant.
                </p>

              </div>

              <User size={20} />

            </div>


            <div className="context-row">

              <span>
                Member
              </span>

              <strong>
  {selectedMember.name ||
    selectedMember.memberName ||
    selectedMember.member_id ||
    selectedMember.memberId ||
    selectedMember.id}
</strong>

            </div>


            <div className="context-row">

              <span>
                Risk Score
              </span>

              <strong>
                {Math.round(
                  selectedMember.riskScore * 100
                )}
                %
              </strong>

            </div>


            <div className="context-row">

              <span>
                Risk Level
              </span>

              <strong>
                {selectedMember.riskLevel}
              </strong>

            </div>


            <div className="context-row">

              <span>
                Primary SDOH
              </span>

              <strong>
                {selectedMember.primarySdoh}
              </strong>

            </div>

          </div>


          <div className="dashboard-card">

            <div className="card-header">

              <div>

                <h2>
                  RAG Context
                </h2>

                <p>
                  Information used for retrieval.
                </p>

              </div>

              <BookOpen size={20} />

            </div>


            <div className="rag-context-item">
              <CheckIcon />
              Member SDOH profile
            </div>

            <div className="rag-context-item">
              <CheckIcon />
              Risk prediction output
            </div>

            <div className="rag-context-item">
              <CheckIcon />
              Intervention ranking
            </div>

            <div className="rag-context-item">
              <CheckIcon />
              Knowledge evidence
            </div>

            {uploadedFile && (

              <div className="rag-context-item">
                <CheckIcon />
                Uploaded document
              </div>

            )}

          </div>

        </div>

      </div>

    </div>
  );
}

const handleSend = async () => {

  if (!input.trim()) return;

  const userMessage = input;

  setMessages((prev) => [
    ...prev,
    {
      role: "user",
      content: userMessage
    }
  ]);

  setInput("");

  try {

    const result = await sendChatMessage(
      userMessage,
      selectedMember
    );

    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        content: result.answer
      }
    ]);

  } catch (error) {

    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        content: "Unable to connect to AI assistant."
      }
    ]);

  }
};

function CheckIcon() {
  return (
    <span className="context-check">
      ✓
    </span>
  );
}


export default RAGAssistant;