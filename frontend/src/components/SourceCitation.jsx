function SourceCitation({ sources = [] }) {

  return (
    <div className="message-sources">

      <div className="source-heading">
        Sources
      </div>

      {sources.map((source, index) => (

        <div key={index}>
          <strong>
            [{index + 1}] {source.title}
          </strong>

          <span>
            {source.location}
          </span>
        </div>

      ))}

    </div>
  );
}

export default SourceCitation;