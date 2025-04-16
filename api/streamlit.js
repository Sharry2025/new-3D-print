const { exec } = require("child_process");

export default async (req, res) => {
  const streamlit = exec(
    "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0",
    { stdio: "inherit" }
  );
  streamlit.stdout.pipe(res);
  streamlit.stderr.pipe(res);
  await new Promise(() => {});
};
